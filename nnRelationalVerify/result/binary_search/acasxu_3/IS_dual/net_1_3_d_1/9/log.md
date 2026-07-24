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
execution time: IAR + LP analysis = 2.00 + 1.59 = 3.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -191.4338822, upper bound: 191.4338822


# Binary Search by BASE starts (time budget: 1196.41 seconds, max iter: 100)

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
Binary search time: 68.62 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1127.79 seconds

## Binary search (step 0) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.6756343, upper bound: 188.5359257
time: 0.55 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1353896, upper bound: 191.1353919
time: 0.60 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.32 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.32
Output dim: 0, lower bound: -179.6756343, upper bound: 188.5359257
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.32
Output dim: 0, lower bound: -191.1353896, upper bound: 191.1353919

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -165.7317810, 169.9891968, -131.9812012, 138.2930145, -304.0247803, 301.9703979
1: -136.9342194, 156.5604401, -109.0557861, 126.4673157, -263.4014893, 265.6161804
2: -192.2410583, 168.2551727, -153.1938171, 138.4591217, -330.7001343, 321.4489746
3: -91.3198547, 188.0007629, -80.2488022, 150.3314209, -241.6512756, 268.2495117
4: -208.4816437, 177.5048828, -166.3941803, 146.1310120, -354.6125488, 343.8990479

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.7747242, upper bound: 177.7747242
time: 0.70 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.7747242, upper bound: 188.5359257
time: 0.61 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -126.3109131, 134.7957306, -138.0393982, 141.9259644, -268.2368774, 272.8351440
1: -104.3817749, 123.2117767, -113.9848785, 129.8824921, -234.2642670, 237.1966400
2: -146.5827942, 135.2793121, -160.1121521, 141.6186523, -288.2014465, 295.3914795
3: -78.7137909, 144.2006378, -81.8182144, 156.6791382, -235.3929138, 226.0188599
4: -159.2684784, 142.8177032, -173.9111633, 149.4060059, -308.6744995, 316.7288818

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.5359257, upper bound: 179.6756343
time: 0.56 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.5359257, upper bound: 191.1353919
time: 0.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.08 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.08
Output dim: 0, lower bound: -177.7747242, upper bound: 177.7747242
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.08
Output dim: 0, lower bound: -177.7747242, upper bound: 188.5359257
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.08
Output dim: 0, lower bound: -188.5359257, upper bound: 179.6756343
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.08
Output dim: 0, lower bound: -188.5359257, upper bound: 191.1353919

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -165.7317810, 169.9891968, -165.7308350, 169.9760437, -335.7078247, 335.7199402
1: -136.9342194, 156.5604401, -136.9331818, 156.5503845, -293.4845886, 293.4936218
2: -192.2410583, 168.2551727, -192.2384796, 168.2447968, -360.4857178, 360.4936523
3: -91.3198547, 188.0007629, -91.3181000, 187.9923859, -279.3122253, 279.3188171
4: -208.4816437, 177.5048828, -208.4785309, 177.4915466, -385.9732056, 385.9833984

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.6129999, upper bound: 177.3927854
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2475535, upper bound: 177.2475539
time: 0.52 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -165.7317810, 169.9891968, -126.0566635, 134.6618042, -300.3935852, 296.0458374
1: -136.9342194, 156.5604401, -104.1840591, 123.0892410, -260.0234070, 260.7445068
2: -192.2410583, 168.2551727, -146.3088684, 135.1582336, -327.3991699, 314.5640259
3: -91.3198547, 188.0007629, -78.6480179, 143.9691772, -235.2890320, 266.6487122
4: -208.4816437, 177.5048828, -158.9604797, 142.6918640, -351.1735229, 336.4653320

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.6129999, upper bound: 188.4561793
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2475535, upper bound: 188.3254275
time: 0.52 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -126.3109131, 134.7957306, -165.7317810, 169.9891968, -296.3001099, 300.5275269
1: -104.3817749, 123.2117767, -136.9342194, 156.5604401, -260.9421997, 260.1459961
2: -146.5827942, 135.2793121, -192.2410583, 168.2551727, -314.8379517, 327.5202942
3: -78.7137909, 144.2006378, -91.3198547, 188.0007629, -266.7145081, 235.5204926
4: -159.2684784, 142.8177032, -208.4816437, 177.5048828, -336.7733765, 351.2993469

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.4561765, upper bound: 179.6546883
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3254247, upper bound: 179.2894972
time: 0.56 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -126.3109131, 134.7957306, -126.3109131, 134.7957306, -261.1066284, 261.1066284
1: -104.3817749, 123.2117767, -104.3817749, 123.2117767, -227.5935364, 227.5935211
2: -146.5827942, 135.2793121, -146.5827942, 135.2793121, -281.8620911, 281.8620911
3: -78.7137909, 144.2006378, -78.7137909, 144.2006378, -222.9144287, 222.9144287
4: -159.2684784, 142.8177032, -159.2684784, 142.8177032, -302.0861816, 302.0861816

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2991944, upper bound: 191.0616678
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3254273, upper bound: 191.0593939
time: 0.58 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.05 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -177.6129999, upper bound: 177.3927854
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -177.2475535, upper bound: 177.2475539
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -177.6129999, upper bound: 188.4561793
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -177.2475535, upper bound: 188.3254275
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -188.4561765, upper bound: 179.6546883
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -188.3254247, upper bound: 179.2894972
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -188.2991944, upper bound: 191.0616678
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -188.3254273, upper bound: 191.0593939

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -182.9138031, 177.0833893, -164.4788361, 169.1156006, -352.0294189, 341.5622253
1: -150.8296814, 163.0941467, -135.9205017, 155.7500916, -306.5797424, 299.0146484
2: -211.2808533, 174.5656891, -190.8140259, 167.4003906, -378.6812439, 365.3796387
3: -95.7939606, 203.4659576, -90.9038773, 186.6641693, -282.4581299, 294.3698425
4: -229.0622559, 184.4309082, -206.8967285, 176.6299591, -405.6921997, 391.3275452

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2352643, upper bound: 177.2352643
time: 0.49 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2352643, upper bound: 177.2475539
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -150.8867340, 158.6611786, -165.7308350, 169.9760437, -320.8627930, 324.3920288
1: -124.7843018, 146.0037994, -136.9331818, 156.5503845, -281.3346863, 282.9369812
2: -175.1371307, 157.1483612, -192.2384796, 168.2447968, -343.3818665, 349.3868408
3: -86.2978363, 172.2207794, -91.3181000, 187.9923859, -274.2901917, 263.5388794
4: -189.8798828, 165.8626709, -208.4785309, 177.4915466, -367.3714294, 374.3411865

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2475539, upper bound: 177.2352643
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2475539, upper bound: 177.2475539
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -182.9138031, 177.0833893, -124.8302689, 133.8360443, -316.7498474, 301.9136658
1: -150.8296814, 163.0941467, -103.1956482, 122.3233337, -273.1530151, 266.2897949
2: -211.2808533, 174.5656891, -144.9039612, 134.4078064, -345.6886597, 319.4696655
3: -95.7939606, 203.4659576, -78.2709198, 142.6600800, -238.4540405, 281.7368774
4: -229.0622559, 184.4309082, -157.3988037, 141.9265137, -370.9887695, 341.8297119

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.2771709, upper bound: 188.2987900
time: 0.50 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.2771709, upper bound: 188.3250221
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -150.8867340, 158.6611786, -126.0566635, 134.6618042, -285.5485229, 284.7178345
1: -124.7843018, 146.0037994, -104.1840591, 123.0892410, -247.8735199, 250.1878510
2: -175.1371307, 157.1483612, -146.3088684, 135.1582336, -310.2953186, 303.4572144
3: -86.2978363, 172.2207794, -78.6480179, 143.9691772, -230.2670135, 250.8687897
4: -189.8798828, 165.8626709, -158.9604797, 142.6918640, -332.5717468, 324.8230896

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.2894976, upper bound: 188.2991944
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.2894977, upper bound: 188.3254276
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -125.0050201, 133.9283295, -182.9138031, 177.0833893, -302.0884094, 316.8421326
1: -103.3319321, 122.4077682, -150.8296814, 163.0941467, -266.4260864, 273.2374268
2: -145.0928650, 134.4912262, -211.2808533, 174.5656891, -319.6585693, 345.7720642
3: -78.3162079, 142.8195190, -95.7939606, 203.4659576, -281.7821655, 238.6134796
4: -157.6112518, 142.0132294, -229.0622559, 184.4309082, -342.0421143, 371.0755005

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2987900, upper bound: 179.2771709
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2987900, upper bound: 179.2771709
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -126.3109131, 134.7957306, -150.8867340, 158.6611786, -284.9721069, 285.6824646
1: -104.3817749, 123.2117767, -124.7843018, 146.0037994, -250.3855438, 247.9960785
2: -146.5827942, 135.2793121, -175.1371307, 157.1483612, -303.7311401, 310.4164124
3: -78.7137909, 144.2006378, -86.2978363, 172.2207794, -250.9345703, 230.4984741
4: -159.2684784, 142.8177032, -189.8798828, 165.8626709, -325.1311646, 332.6975708

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2991916, upper bound: 179.2894976
time: 0.53 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2991916, upper bound: 179.2894976
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -146.9546051, 144.1370697, -125.0050201, 133.9283295, -280.8829041, 269.1420593
1: -121.2054749, 131.9143066, -103.3319321, 122.4077682, -243.6132202, 235.2462463
2: -169.7323761, 143.6659851, -145.0928650, 134.4912262, -304.2235718, 288.7587891
3: -83.4981003, 164.1896667, -78.3162079, 142.8195190, -226.3176117, 242.5058746
4: -184.1379089, 151.7879486, -157.6112518, 142.0132294, -326.1511230, 309.3992004

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0334846, upper bound: 191.0334846
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0334846, upper bound: 191.0593914
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -109.4849014, 122.1597290, -126.3109131, 134.7957306, -244.2806244, 248.4706268
1: -90.5823822, 111.4376678, -104.3817749, 123.2117767, -213.7941132, 215.8194122
2: -127.0952301, 124.0488815, -146.5827942, 135.2793121, -262.3745422, 270.6316833
3: -73.2953796, 126.1313477, -78.7137909, 144.2006378, -217.4960175, 204.8451385
4: -138.1383820, 131.0992584, -159.2684784, 142.8177032, -280.9560852, 290.3677368

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -172.4471712, upper bound: 179.9878011
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0071620, upper bound: 191.0064387
time: 0.59 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.14 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -177.2352643, upper bound: 177.2352643
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -177.2352643, upper bound: 177.2475539
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -177.2475539, upper bound: 177.2352643
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -177.2475539, upper bound: 177.2475539
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -179.2771709, upper bound: 188.2987900
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -179.2771709, upper bound: 188.3250221
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -179.2894976, upper bound: 188.2991944
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -179.2894977, upper bound: 188.3254276
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -188.2987900, upper bound: 179.2771709
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -188.2987900, upper bound: 179.2771709
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -188.2991916, upper bound: 179.2894976
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -188.2991916, upper bound: 179.2894976
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -191.0334846, upper bound: 191.0334846
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -191.0334846, upper bound: 191.0593914
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -172.4471712, upper bound: 179.9878011
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -191.0071620, upper bound: 191.0064387

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -182.9138031, 177.0833893, -182.9076538, 176.9971466, -359.9109497, 359.9910278
1: -150.8296814, 163.0941467, -150.8231964, 163.0276184, -313.8572998, 313.9173584
2: -211.2808533, 174.5656891, -211.2644501, 174.4967804, -385.7776184, 385.8301086
3: -95.7939606, 203.4659576, -95.7824173, 203.4107056, -299.2046509, 299.2483521
4: -229.0622559, 184.4309082, -229.0425720, 184.3425751, -413.4048462, 413.4733887

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.1841121, upper bound: 176.8334547
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9048567, upper bound: 175.8481590
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -182.9138031, 177.0833893, -150.8867340, 158.6611786, -341.5749817, 327.9701233
1: -150.8296814, 163.0941467, -124.7843018, 146.0037994, -296.8334351, 287.8784485
2: -211.2808533, 174.5656891, -175.1371307, 157.1483612, -368.4291992, 349.7027588
3: -95.7939606, 203.4659576, -86.2978363, 172.2207794, -268.0147400, 289.7637939
4: -229.0622559, 184.4309082, -189.8798828, 165.8626709, -394.9249268, 374.3107605

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9008758, upper bound: 173.4424160
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.3079463, upper bound: 177.0588510
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -150.8867340, 158.6611786, -182.9076538, 176.9971466, -327.8838806, 341.5688477
1: -124.7843018, 146.0037994, -150.8231964, 163.0276184, -287.8119202, 296.8269348
2: -175.1371307, 157.1483612, -211.2644501, 174.4967804, -349.6339111, 368.4128113
3: -86.2978363, 172.2207794, -95.7824173, 203.4107056, -289.7085266, 268.0031738
4: -189.8798828, 165.8626709, -229.0425720, 184.3425751, -374.2224731, 394.9051514

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2472508, upper bound: 177.2335074
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -173.4424134, upper bound: 175.5183225
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9085858, upper bound: 176.8980233
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -150.8867340, 158.6611786, -150.8867340, 158.6611786, -309.5479126, 309.5479126
1: -124.7843018, 146.0037994, -124.7843018, 146.0037994, -270.7880554, 270.7880249
2: -175.1371307, 157.1483612, -175.1371307, 157.1483612, -332.2854614, 332.2854614
3: -86.2978363, 172.2207794, -86.2978363, 172.2207794, -258.5186157, 258.5186157
4: -189.8798828, 165.8626709, -189.8798828, 165.8626709, -355.7425537, 355.7425537

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -173.4424160, upper bound: 175.5288859
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9085863, upper bound: 176.9058406
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -182.9138031, 177.0833893, -146.8650665, 144.0907135, -327.0045166, 323.9484558
1: -150.8296814, 163.0941467, -121.1358032, 131.8713379, -282.7010193, 284.2299500
2: -211.2808533, 174.5656891, -169.6356354, 143.6236267, -354.9044800, 344.2012329
3: -95.7939606, 203.4659576, -83.4746933, 164.1075439, -259.9014893, 286.9406433
4: -229.0622559, 184.4309082, -184.0289612, 151.7441101, -380.8063660, 368.4598389

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.3043024, upper bound: 176.3912616
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0113014, upper bound: 175.9834638
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -182.9138031, 177.0833893, -109.2646790, 122.0444794, -304.9582825, 286.3480835
1: -150.8296814, 163.0941467, -90.4099121, 111.3319626, -262.1616516, 253.5040588
2: -211.2808533, 174.5656891, -126.8569641, 123.9463348, -335.2271729, 301.4225769
3: -95.7939606, 203.4659576, -73.2382278, 125.9308929, -221.7248535, 276.7041931
4: -229.0622559, 184.4309082, -137.8677826, 130.9930878, -360.0553589, 322.2986755

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.7187146, upper bound: 172.1567543
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.4653525, upper bound: 188.2799185
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -150.8867340, 158.6611786, -146.8650665, 144.0907135, -294.9774475, 305.5262451
1: -124.7843018, 146.0037994, -121.1358032, 131.8713379, -256.6556396, 267.1395264
2: -175.1371307, 157.1483612, -169.6356354, 143.6236267, -318.7607422, 326.7839355
3: -86.2978363, 172.2207794, -83.4746933, 164.1075439, -250.4053802, 255.6954651
4: -189.8798828, 165.8626709, -184.0289612, 151.7441101, -341.6239929, 349.8916321

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.2891961, upper bound: 188.2985994
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.2364790, upper bound: 188.2179376
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.4576277, upper bound: 184.5574171
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.0662526, upper bound: 188.1156089
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -150.8867340, 158.6611786, -109.2646790, 122.0444794, -272.9312134, 267.9258423
1: -124.7843018, 146.0037994, -90.4099121, 111.3319626, -236.1162720, 236.4136963
2: -175.1371307, 157.1483612, -126.8569641, 123.9463348, -299.0834351, 284.0053101
3: -86.2978363, 172.2207794, -73.2382278, 125.9308929, -212.2287292, 245.4590149
4: -189.8798828, 165.8626709, -137.8677826, 130.9930878, -320.8729858, 303.7304688

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.3193548, upper bound: 171.9674624
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.0662536, upper bound: 188.1348402
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -146.9546051, 144.1370697, -182.9138031, 177.0833893, -324.0379944, 327.0508728
1: -121.2054749, 131.9143066, -150.8296814, 163.0941467, -284.2996216, 282.7439880
2: -169.7323761, 143.6659851, -211.2808533, 174.5656891, -344.2980347, 354.9468079
3: -83.4981003, 164.1896667, -95.7939606, 203.4659576, -286.9640503, 259.9835815
4: -184.1379089, 151.7879486, -229.0622559, 184.4309082, -368.5688171, 380.8502197

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3912616, upper bound: 177.3043024
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9834638, upper bound: 176.0113014
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -109.4849014, 122.1597290, -182.9138031, 177.0833893, -286.5682983, 305.0735474
1: -90.5823822, 111.4376678, -150.8296814, 163.0941467, -253.6765289, 262.2673035
2: -127.0952301, 124.0488815, -211.2808533, 174.5656891, -301.6609192, 335.3297424
3: -73.2953796, 126.1313477, -95.7939606, 203.4659576, -276.7613525, 221.9253082
4: -138.1383820, 131.0992584, -229.0622559, 184.4309082, -322.5691833, 360.1614990

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.4596035, upper bound: 175.7832481
time: 0.58 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2561598, upper bound: 179.4653525
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -146.9546051, 144.1370697, -150.8867340, 158.6611786, -305.6157837, 295.0237732
1: -121.2054749, 131.9143066, -124.7843018, 146.0037994, -267.2092590, 256.6986084
2: -169.7323761, 143.6659851, -175.1371307, 157.1483612, -326.8807068, 318.8030701
3: -83.4981003, 164.1896667, -86.2978363, 172.2207794, -255.7188721, 250.4874878
4: -184.1379089, 151.7879486, -189.8798828, 165.8626709, -350.0005493, 341.6678467

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2179376, upper bound: 179.2364790
time: 0.55 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.0186689, upper bound: 175.4576277
time: 0.56 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.1156089, upper bound: 179.0662526
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -109.4849014, 122.1597290, -150.8867340, 158.6611786, -268.1460876, 273.0464172
1: -90.5823822, 111.4376678, -124.7843018, 146.0037994, -236.5861816, 236.2219543
2: -127.0952301, 124.0488815, -175.1371307, 157.1483612, -284.2435913, 299.1860046
3: -73.2953796, 126.1313477, -86.2978363, 172.2207794, -245.5161591, 212.4291840
4: -138.1383820, 131.0992584, -189.8798828, 165.8626709, -304.0009766, 320.9791260

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.3093388, upper bound: 175.4171965
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.1156076, upper bound: 179.0656352
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -146.9546051, 144.1370697, -146.9546051, 144.1370697, -291.0916748, 291.0916748
1: -121.2054749, 131.9143066, -121.2054749, 131.9143066, -253.1197815, 253.1197815
2: -169.7323761, 143.6659851, -169.7323761, 143.6659851, -313.3982849, 313.3983154
3: -83.4981003, 164.1896667, -83.4981003, 164.1896667, -247.6877594, 247.6877594
4: -184.1379089, 151.7879486, -184.1379089, 151.7879486, -335.9258423, 335.9258423

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.6859833, upper bound: 178.3366485
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0042211, upper bound: 176.0622054
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -146.9546051, 144.1370697, -109.4849014, 122.1597290, -269.1143188, 253.6219788
1: -121.2054749, 131.9143066, -90.5823822, 111.4376678, -232.6431274, 222.4966736
2: -169.7323761, 143.6659851, -127.0952301, 124.0488815, -293.7812500, 270.7612305
3: -83.4981003, 164.1896667, -73.2953796, 126.1313477, -209.6294403, 237.4850464
4: -184.1379089, 151.7879486, -138.1383820, 131.0992584, -315.2371826, 289.9262695

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.5887587, upper bound: 174.2160659
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.9716494, upper bound: 191.0060342
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -91.4743271, 112.7514648, -123.9862747, 133.2964630, -224.7707520, 236.7377319
1: -74.9706650, 101.6991348, -102.4597321, 121.8221283, -196.7927856, 204.1588745
2: -105.2589417, 115.1343307, -143.8735199, 134.0278015, -239.2867432, 259.0078430
3: -68.8409653, 104.0475159, -78.0798416, 141.6001129, -210.4410706, 182.1273499
4: -115.4177475, 121.8774719, -156.3420410, 141.5175629, -256.9353027, 278.2194519

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -172.3398954, upper bound: 179.8732649
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -172.3398980, upper bound: 179.9878019
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -105.0934448, 119.2186661, -126.3109131, 134.7957306, -239.8891296, 245.5295715
1: -86.9803162, 108.7176056, -104.3817749, 123.2117767, -210.1920929, 213.0993500
2: -122.0204010, 120.9229431, -146.5827942, 135.2793121, -257.2997131, 267.5057373
3: -71.7856598, 121.2669067, -78.7137909, 144.2006378, -215.9862823, 199.9806824
4: -132.6455536, 127.7139664, -159.2684784, 142.8177032, -275.4632568, 286.9824524

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0071620, upper bound: 191.0020547
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0068829, upper bound: 191.0020547
time: 0.59 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.33 seconds
IS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -176.1841121, upper bound: 176.8334547
IS_A1_B1_A1_B1_B2, status: Status.VERIFIED, split count: 5, time: 3.33
Output dim: 0, lower bound: -175.9048567, upper bound: 175.8481590
IS_A1_B1_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 3.33
Output dim: 0, lower bound: -175.9008758, upper bound: 173.4424160
IS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -177.3079463, upper bound: 177.0588510
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.33
Output dim: 0, lower bound: -173.4424134, upper bound: 175.5183225
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -176.9085858, upper bound: 176.8980233
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.33
Output dim: 0, lower bound: -173.4424160, upper bound: 175.5288859
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -176.9085863, upper bound: 176.9058406
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -177.3043024, upper bound: 176.3912616
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.33
Output dim: 0, lower bound: -176.0113014, upper bound: 175.9834638
IS_A1_B2_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 3.33
Output dim: 0, lower bound: -175.7187146, upper bound: 172.1567543
IS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -179.4653525, upper bound: 188.2799185
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -175.4576277, upper bound: 184.5574171
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -179.0662526, upper bound: 188.1156089
IS_A1_B2_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 3.33
Output dim: 0, lower bound: -175.3193548, upper bound: 171.9674624
IS_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -179.0662536, upper bound: 188.1348402
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -176.3912616, upper bound: 177.3043024
IS_A2_B1_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 3.33
Output dim: 0, lower bound: -175.9834638, upper bound: 176.0113014
IS_A2_B1_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 3.33
Output dim: 0, lower bound: -171.4596035, upper bound: 175.7832481
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -188.2561598, upper bound: 179.4653525
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -184.0186689, upper bound: 175.4576277
IS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -188.1156089, upper bound: 179.0662526
IS_A2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 3.33
Output dim: 0, lower bound: -171.3093388, upper bound: 175.4171965
IS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -188.1156076, upper bound: 179.0656352
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -188.6859833, upper bound: 178.3366485
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.33
Output dim: 0, lower bound: -176.0042211, upper bound: 176.0622054
IS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -184.5887587, upper bound: 174.2160659
IS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -190.9716494, upper bound: 191.0060342
IS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -172.3398954, upper bound: 179.8732649
IS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -172.3398980, upper bound: 179.9878019
IS_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -191.0071620, upper bound: 191.0020547
IS_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -191.0068829, upper bound: 191.0020547

## BFS IS instance: IS_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -182.9138031, 177.0833893, -177.3173370, 173.0451965, -355.9589844, 354.4007263
1: -150.8296814, 163.0941467, -146.2355804, 159.3224487, -310.1521301, 309.3297119
2: -211.2808533, 174.5656891, -204.8420105, 170.6755066, -381.9563599, 379.4077148
3: -95.7939606, 203.4659576, -93.9955902, 197.5093689, -293.3033142, 297.4615479
4: -229.0622559, 184.4309082, -222.0899048, 180.2908630, -409.3531189, 406.5207520

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.2527551, upper bound: 172.5563935
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.1000190, upper bound: 176.9681099
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -182.9138031, 177.0833893, -146.3338928, 155.7396545, -338.6534424, 323.4172974
1: -150.8296814, 163.0941467, -121.0181732, 143.3005981, -294.1302795, 284.1123047
2: -211.2808533, 174.5656891, -169.8362579, 154.3646698, -365.6455078, 344.4019165
3: -95.7939606, 203.4659576, -84.6786346, 167.1723022, -262.9662476, 288.1445923
4: -229.0622559, 184.4309082, -184.1595917, 162.9274750, -391.9897461, 368.5904846

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.3044594, upper bound: 177.0585112
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9631287, upper bound: 176.0585341
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.6980359, upper bound: 175.7509384
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -146.3338928, 155.7396545, -182.9076538, 176.9971466, -323.3310547, 338.6473083
1: -121.0181732, 143.3005981, -150.8231964, 163.0276184, -284.0457764, 294.1237793
2: -169.8362579, 154.3646698, -211.2644501, 174.4967804, -344.3330383, 365.6291199
3: -84.6786346, 167.1723022, -95.7824173, 203.4107056, -288.0893555, 262.9547119
4: -184.1595917, 162.9274750, -229.0425720, 184.3425751, -368.5021667, 391.9700012

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.0585112, upper bound: 177.3044594
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.0585341, upper bound: 176.9631287
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.7509384, upper bound: 175.6980359
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -146.3338928, 155.7396545, -150.8867340, 158.6611786, -304.9950562, 306.6264038
1: -121.0181732, 143.3005981, -124.7843018, 146.0037994, -267.0219421, 268.0848999
2: -169.8362579, 154.3646698, -175.1371307, 157.1483612, -326.9845886, 329.5017395
3: -84.6786346, 167.1723022, -86.2978363, 172.2207794, -256.8994141, 253.4701385
4: -184.1595917, 162.9274750, -189.8798828, 165.8626709, -350.0222778, 352.8073730

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9082466, upper bound: 176.9050993
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9047597, upper bound: 176.9047600
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -177.3217163, 173.1061249, -146.8650665, 144.0907135, -321.4124146, 319.9710999
1: -146.2401733, 159.3694153, -121.1358032, 131.8713379, -278.1115112, 280.5051880
2: -204.8536377, 170.7242126, -169.6356354, 143.6236267, -348.4772644, 340.3597717
3: -94.0037155, 197.5484619, -83.4746933, 164.1075439, -258.1112366, 281.0231323
4: -222.1038513, 180.3532715, -184.0289612, 151.7441101, -373.8479309, 364.3822327

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.6978837, upper bound: 173.1560670
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.8103332, upper bound: 169.0455043
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.0473460, upper bound: 176.1513841
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -182.9138031, 177.0833893, -105.0801926, 119.2118301, -302.1256409, 282.1635742
1: -150.8296814, 163.0941467, -86.9700012, 108.7113190, -259.5409546, 250.0641479
2: -211.2808533, 174.5656891, -122.0060883, 120.9168701, -332.1977234, 296.5717773
3: -95.7939606, 203.4659576, -71.7822876, 121.2549667, -217.0489197, 275.2482300
4: -229.0622559, 184.4309082, -132.6293030, 127.7076950, -356.7699585, 317.0602112

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.4641248, upper bound: 188.2799212
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.4275135, upper bound: 188.2052321
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.4653526, upper bound: 188.2799212
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.4653525, upper bound: 188.1752826
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -132.5038147, 148.8401642, -144.6946869, 142.6764679, -275.1802979, 293.5347900
1: -109.3140106, 137.3092957, -119.3478470, 130.5563965, -239.8704071, 256.6571350
2: -153.3966522, 148.6413116, -167.1113281, 142.4461975, -295.8428345, 315.7526245
3: -81.1663208, 152.0792236, -82.8711929, 161.6640015, -242.8303223, 234.9503937
4: -166.8361816, 156.6905060, -181.2968750, 150.5253448, -317.3614807, 337.9873352

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.4576277, upper bound: 184.5574150
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.4131630, upper bound: 184.4896397
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -174.9159320, upper bound: 180.4211776
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -171.0984186, upper bound: 178.9310931
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -146.3338928, 155.7396545, -146.8650665, 144.0907135, -290.4246216, 302.6047058
1: -121.0181732, 143.3005981, -121.1358032, 131.8713379, -252.8895111, 264.4364014
2: -169.8362579, 154.3646698, -169.6356354, 143.6236267, -313.4598694, 324.0002136
3: -84.6786346, 167.1723022, -83.4746933, 164.1075439, -248.7861786, 250.6469879
4: -184.1595917, 162.9274750, -184.0289612, 151.7441101, -335.9036865, 346.9564209

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.0674205, upper bound: 188.1137030
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.0158596, upper bound: 188.0334695
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.1724239, upper bound: 187.4439499
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.8301747, upper bound: 175.7494022
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -150.8867340, 158.6611786, -105.0801926, 119.2118301, -270.0985413, 263.7413635
1: -124.7843018, 146.0037994, -86.9700012, 108.7113190, -233.4956055, 232.9738007
2: -175.1371307, 157.1483612, -122.0060883, 120.9168701, -296.0539856, 279.1544495
3: -86.2978363, 172.2207794, -71.7822876, 121.2549667, -207.5527802, 244.0030670
4: -189.8798828, 165.8626709, -132.6293030, 127.7076950, -317.5875854, 298.4919739

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.0659156, upper bound: 188.1345603
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.0646566, upper bound: 188.1345603
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -146.9546051, 144.1370697, -177.3217163, 173.1061249, -320.0607300, 321.4588013
1: -121.2054749, 131.9143066, -146.2401733, 159.3694153, -280.5748596, 278.1544800
2: -169.7323761, 143.6659851, -204.8536377, 170.7242126, -340.4565430, 348.5196228
3: -83.4981003, 164.1896667, -94.0037155, 197.5484619, -281.0465393, 258.1933289
4: -184.1379089, 151.7879486, -222.1038513, 180.3532715, -364.4911804, 373.8917847

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -173.1560670, upper bound: 175.6978837
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -169.0455043, upper bound: 174.8103332
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.1513840, upper bound: 177.0473460
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -105.0934448, 119.2186661, -182.9138031, 177.0833893, -282.1768188, 302.1324768
1: -86.9803162, 108.7176056, -150.8296814, 163.0941467, -250.0744629, 259.5472717
2: -122.0204010, 120.9229431, -211.2808533, 174.5656891, -296.5860901, 332.2037964
3: -71.7856598, 121.2669067, -95.7939606, 203.4659576, -275.2516174, 217.0608673
4: -132.6455536, 127.7139664, -229.0622559, 184.4309082, -317.0764465, 356.7762146

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2799184, upper bound: 179.4641247
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2052321, upper bound: 179.4275135
time: 0.53 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2799184, upper bound: 179.4653525
time: 0.56 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.1752826, upper bound: 179.4653525
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -144.7281494, 142.6937256, -132.5038147, 148.8401642, -293.5682678, 275.1975098
1: -119.3738785, 130.5724030, -109.3140106, 137.3092957, -256.6831665, 239.8864136
2: -167.1474609, 142.4619598, -153.3966522, 148.6413116, -315.7887573, 295.8585815
3: -82.8798676, 161.6947021, -81.1663208, 152.0792236, -234.9590912, 242.8610229
4: -181.3374939, 150.5416412, -166.8361816, 156.6905060, -338.0280151, 317.3778076

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.5574150, upper bound: 175.4576277
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.4896407, upper bound: 175.4131630
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_B2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.4211776, upper bound: 174.9159320
time: 0.59 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.9310931, upper bound: 171.0984186
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -146.9546051, 144.1370697, -146.3338928, 155.7396545, -302.6942749, 290.4709473
1: -121.2054749, 131.9143066, -121.0181732, 143.3005981, -264.5060730, 252.9324799
2: -169.7323761, 143.6659851, -169.8362579, 154.3646698, -324.0970154, 313.5021667
3: -83.4981003, 164.1896667, -84.6786346, 167.1723022, -250.6703796, 248.8683014
4: -184.1379089, 151.7879486, -184.1595917, 162.9274750, -347.0653687, 335.9475403

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.1137030, upper bound: 179.0674205
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.0334695, upper bound: 179.0158596
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4439499, upper bound: 178.1724239
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.7494022, upper bound: 175.8301747
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -105.0934448, 119.2186661, -150.8867340, 158.6611786, -263.7546082, 270.1054077
1: -86.9803162, 108.7176056, -124.7843018, 146.0037994, -232.9841156, 233.5019073
2: -122.0204010, 120.9229431, -175.1371307, 157.1483612, -279.1687622, 296.0600281
3: -71.7856598, 121.2669067, -86.2978363, 172.2207794, -244.0064392, 207.5647430
4: -132.6455536, 127.7139664, -189.8798828, 165.8626709, -298.5082397, 317.5938416

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_A2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.1491075, upper bound: 179.0652650
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.1491082, upper bound: 179.0646842
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -140.6025085, 139.6843872, -146.9546051, 144.1370697, -284.7395630, 286.6389771
1: -115.9742508, 127.7212601, -121.2054749, 131.9143066, -247.8885498, 248.9267273
2: -162.3865051, 139.2944336, -169.7323761, 143.6659851, -306.0524902, 309.0267944
3: -81.5001144, 157.3161316, -83.4981003, 164.1896667, -245.6897888, 240.8142395
4: -176.2256470, 147.1510620, -184.1379089, 151.7879486, -328.0136108, 331.2889709

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.6104674, upper bound: 178.2910054
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0042211, upper bound: 176.0622053
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0042211, upper bound: 176.0622054
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -144.7281494, 142.6937256, -91.4743271, 112.7514648, -257.4796143, 234.1680298
1: -119.3738785, 130.5724030, -74.9706650, 101.6991348, -221.0730133, 205.5430603
2: -167.1474609, 142.4619598, -105.2589417, 115.1343307, -282.2817993, 247.7209015
3: -82.8798676, 161.6947021, -68.8409653, 104.0475159, -186.9273834, 230.5356445
4: -181.3374939, 150.5416412, -115.4177475, 121.8774719, -303.2149658, 265.9593811

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.3063736, upper bound: 174.2160636
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.6653737, upper bound: 173.0405124
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.3063744, upper bound: 174.2160636
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -146.9546051, 144.1370697, -105.0934448, 119.2186661, -266.1732788, 249.2304840
1: -121.2054749, 131.9143066, -86.9803162, 108.7176056, -229.9230652, 218.8946228
2: -169.7323761, 143.6659851, -122.0204010, 120.9229431, -290.6552734, 265.6863708
3: -83.4981003, 164.1896667, -71.7856598, 121.2669067, -204.7649994, 235.9753265
4: -184.1379089, 151.7879486, -132.6455536, 127.7139664, -311.8518677, 284.4335022

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.9715486, upper bound: 191.0060334
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8751565, upper bound: 190.8846086
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.9716497, upper bound: 191.0060334
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -91.4743271, 112.7514648, -144.7281494, 142.6937256, -234.1680603, 257.4796143
1: -74.9706650, 101.6991348, -119.3738785, 130.5724030, -205.5430603, 221.0730133
2: -105.2589417, 115.1343307, -167.1474609, 142.4619598, -247.7209015, 282.2817993
3: -68.8409653, 104.0475159, -82.8798676, 161.6947021, -230.5356445, 186.9273834
4: -115.4177475, 121.8774719, -181.3374939, 150.5416412, -265.9593811, 303.2149658

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -172.3398954, upper bound: 179.8732649
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -171.8573991, upper bound: 179.3392363
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -170.4244641, upper bound: 177.5305761
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -91.4743271, 112.7514648, -107.1884460, 120.8012085, -212.2754974, 219.9399109
1: -74.9706650, 101.6991348, -88.6859131, 110.0723114, -185.0429382, 190.3850403
2: -105.2589417, 115.1343307, -124.4192505, 122.8460541, -228.1049957, 239.5535583
3: -68.8409653, 104.0475159, -72.6748505, 123.5625992, -192.4035645, 176.7223663
4: -115.4177475, 121.8774719, -135.2475128, 129.8513489, -245.2691040, 257.1250000

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -172.3398980, upper bound: 179.9709471
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -169.1123969, upper bound: 178.6005271
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -170.4244576, upper bound: 179.4081194
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -105.0934448, 119.2186661, -114.0290833, 127.2539062, -232.3473511, 233.2477417
1: -86.9803162, 108.7176056, -94.4107437, 116.1878967, -203.1682129, 203.1283264
2: -122.0204010, 120.9229431, -132.5791931, 128.3272858, -250.3476868, 253.5021362
3: -71.7856598, 121.2669067, -75.1249390, 131.1520691, -202.9377289, 196.3918457
4: -132.6455536, 127.7139664, -143.9013977, 135.5662689, -268.2118225, 271.6153564

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0063637, upper bound: 190.9715639
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0063644, upper bound: 191.0006429
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -105.0934448, 119.2186661, -150.0061340, 152.2707062, -257.3641052, 269.2247925
1: -86.9803162, 108.7176056, -123.4277039, 139.9473572, -226.9276733, 232.1452942
2: -122.0204010, 120.9229431, -173.8018036, 152.2116852, -274.2320862, 294.7247314
3: -71.7856598, 121.2669067, -86.5325623, 170.2913971, -242.0770569, 207.7994690
4: -132.6455536, 127.7139664, -189.6269226, 160.3744354, -293.0199890, 317.3408813

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0068829, upper bound: 191.0020547
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -171.6226358, upper bound: 191.0020547
time: 0.66 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.84 seconds
IS_A1_B1_A1_B1_B1_B1, status: Status.VERIFIED, split count: 6, time: 3.84
Output dim: 0, lower bound: -174.2527551, upper bound: 172.5563935
IS_A1_B1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -176.1000190, upper bound: 176.9681099
IS_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -176.9631287, upper bound: 176.0585341
IS_A1_B1_A1_B2_B2_A2, status: Status.VERIFIED, split count: 6, time: 3.84
Output dim: 0, lower bound: -175.6980359, upper bound: 175.7509384
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -176.0585341, upper bound: 176.9631287
IS_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.84
Output dim: 0, lower bound: -175.7509384, upper bound: 175.6980359
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -176.9082466, upper bound: 176.9050993
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -176.9047597, upper bound: 176.9047600
IS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.84
Output dim: 0, lower bound: -174.8103332, upper bound: 169.0455043
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -177.0473460, upper bound: 176.1513841
IS_A1_B2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -179.4653526, upper bound: 188.2799212
IS_A1_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -179.4653525, upper bound: 188.1752826
IS_A1_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -174.9159320, upper bound: 180.4211776
IS_A1_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -171.0984186, upper bound: 178.9310931
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -178.1724239, upper bound: 187.4439499
IS_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.84
Output dim: 0, lower bound: -175.8301747, upper bound: 175.7494022
IS_A1_B2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -179.0659156, upper bound: 188.1345603
IS_A1_B2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -179.0646566, upper bound: 188.1345603
IS_A2_B1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.84
Output dim: 0, lower bound: -169.0455043, upper bound: 174.8103332
IS_A2_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -176.1513840, upper bound: 177.0473460
IS_A2_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -188.2799184, upper bound: 179.4653525
IS_A2_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -188.1752826, upper bound: 179.4653525
IS_A2_B1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -180.4211776, upper bound: 174.9159320
IS_A2_B1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -178.9310931, upper bound: 171.0984186
IS_A2_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -187.4439499, upper bound: 178.1724239
IS_A2_B1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 3.84
Output dim: 0, lower bound: -175.7494022, upper bound: 175.8301747
IS_A2_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -188.1491075, upper bound: 179.0652650
IS_A2_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -188.1491082, upper bound: 179.0646842
IS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.84
Output dim: 0, lower bound: -176.0042211, upper bound: 176.0622053
IS_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.84
Output dim: 0, lower bound: -176.0042211, upper bound: 176.0622054
IS_A2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -180.6653737, upper bound: 173.0405124
IS_A2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -185.3063744, upper bound: 174.2160636
IS_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -186.8751565, upper bound: 190.8846086
IS_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -190.9716497, upper bound: 191.0060334
IS_A2_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -171.8573991, upper bound: 179.3392363
IS_A2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -170.4244641, upper bound: 177.5305761
IS_A2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -169.1123969, upper bound: 178.6005271
IS_A2_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -170.4244576, upper bound: 179.4081194
IS_A2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -191.0063637, upper bound: 190.9715639
IS_A2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -191.0063644, upper bound: 191.0006429
IS_A2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -191.0068829, upper bound: 191.0020547
IS_A2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 0, lower bound: -171.6226358, upper bound: 191.0020547

## BFS IS instance: IS_A1_B1_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -182.9138031, 177.0833893, -172.6314697, 170.0600433, -352.9738464, 349.7148438
1: -150.8296814, 163.0941467, -142.3586884, 156.5465393, -307.3762207, 305.4528198
2: -211.2808533, 174.5656891, -199.3724976, 167.8279877, -379.1088257, 373.9381714
3: -95.7939606, 203.4659576, -92.2895203, 192.2710571, -288.0650024, 295.7554932
4: -229.0622559, 184.4309082, -216.1923218, 177.2752991, -406.3375549, 400.6231995

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.1000190, upper bound: 176.9681099
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.1000190, upper bound: 176.9681099
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -177.3217163, 173.1061249, -146.3338928, 155.7396545, -333.0613708, 319.4400024
1: -146.2401733, 159.3694153, -121.0181732, 143.3005981, -289.5407715, 280.3875732
2: -204.8536377, 170.7242126, -169.8362579, 154.3646698, -359.2182617, 340.5604553
3: -94.0037155, 197.5484619, -84.6786346, 167.1723022, -261.1759949, 282.2271118
4: -222.1038513, 180.3532715, -184.1595917, 162.9274750, -385.0313110, 364.5128784

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2553932, upper bound: 175.3372759
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.5657933, upper bound: 175.4525573
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.6980359, upper bound: 175.7509384
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.6980359, upper bound: 175.7509384
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -146.3338928, 155.7396545, -177.3173370, 173.0451965, -319.3790894, 333.0570068
1: -121.0181732, 143.3005981, -146.2355804, 159.3224487, -280.3406372, 289.5361633
2: -169.8362579, 154.3646698, -204.8420105, 170.6755066, -340.5117493, 359.2066650
3: -84.6786346, 167.1723022, -93.9955902, 197.5093689, -282.1879883, 261.1679077
4: -184.1595917, 162.9274750, -222.0899048, 180.2908630, -364.4504089, 385.0173645

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.3372759, upper bound: 176.2553932
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.4525573, upper bound: 176.5657933
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.7509384, upper bound: 175.6980359
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.7509384, upper bound: 175.6980359
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -146.3338928, 155.7396545, -140.5636597, 152.4796143, -298.8134460, 296.3033142
1: -121.0181732, 143.3005981, -116.3886566, 140.2610321, -261.2792053, 259.6892700
2: -169.8362579, 154.3646698, -163.3916016, 151.2188873, -321.0551453, 317.7562256
3: -84.6786346, 167.1723022, -83.0665359, 161.3816833, -246.0603180, 250.2388306
4: -184.1595917, 162.9274750, -177.0340424, 159.6253052, -343.7849121, 339.9614868

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9047597, upper bound: 176.9047600
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9047597, upper bound: 176.9047600
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -146.3338928, 155.7396545, -169.3839722, 172.7578125, -319.0916443, 325.1236267
1: -121.0181732, 143.3005981, -139.7002258, 159.7275085, -280.7456360, 283.0008240
2: -169.8362579, 154.3646698, -196.4113464, 172.1500092, -341.9862366, 350.7759705
3: -84.6786346, 167.1723022, -92.4798431, 192.8359680, -277.5145874, 259.6521301
4: -184.1595917, 162.9274750, -213.3372040, 181.2526245, -365.4122314, 376.2646790

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9047597, upper bound: 176.9047600
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9047597, upper bound: 176.9047600
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -177.3217163, 173.1061249, -141.8547516, 140.8447418, -318.1664429, 314.9608765
1: -146.2401733, 159.3694153, -116.9957733, 128.8614807, -275.1016541, 276.3651733
2: -204.8536377, 170.7242126, -163.8165283, 140.1550446, -345.0086670, 334.5407104
3: -94.0037155, 197.5484619, -81.7145462, 158.5599518, -252.5636597, 279.2629395
4: -222.1038513, 180.3532715, -177.7744446, 147.9893799, -370.0932312, 358.1277161

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.4226257, upper bound: 172.9173710
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3304260, upper bound: 174.2386387
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.6431994, upper bound: 170.8978894
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -170.8855283, upper bound: 170.4864009
time: 0.49 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -182.5645752, 176.8494873, -81.8415833, 104.8546906, -287.4192505, 258.6910706
1: -150.5458374, 162.8777771, -68.0090408, 94.7844849, -245.3303223, 230.8868103
2: -210.8809204, 174.3390656, -95.3412018, 107.1040955, -317.9850159, 269.6802673
3: -95.6772766, 203.0880432, -64.9595795, 96.7800903, -192.4573212, 268.0476074
4: -228.6232605, 184.1927490, -103.6640930, 113.4794769, -342.1026917, 287.8568420

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.3318889, upper bound: 187.9861942
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.4275135, upper bound: 188.2052321
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.6764674, upper bound: 182.2613831
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -173.3084914, upper bound: 179.9816063
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -182.9138031, 177.0833893, -98.1761932, 114.4870758, -297.4008789, 275.2595520
1: -150.8296814, 163.0941467, -81.2362823, 104.1438599, -254.9734497, 244.3304291
2: -211.2808533, 174.5656891, -113.9444351, 116.7703171, -328.0511475, 288.5100708
3: -95.7939606, 203.4659576, -69.7811661, 113.5294724, -209.3234100, 273.2471313
4: -229.0622559, 184.4309082, -123.9752121, 123.4040375, -352.4663086, 308.4060974

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.4275135, upper bound: 188.0981194
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.3318889, upper bound: 187.8974142
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.2424479, upper bound: 186.7143558
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -171.5006884, upper bound: 184.0297551
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -171.5006884, upper bound: 184.5912570
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -120.8671112, 139.6277618, -144.6946869, 142.6764679, -263.5435486, 284.3224487
1: -99.6032562, 128.4012909, -119.3478470, 130.5563965, -230.1596527, 247.7491455
2: -139.7168884, 139.6352539, -167.1113281, 142.4461975, -282.1630859, 306.7465820
3: -77.4247513, 138.7147980, -82.8711929, 161.6640015, -239.0887451, 221.5859680
4: -152.2470703, 147.2355194, -181.2968750, 150.5253448, -302.7723999, 328.5323486

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -174.9115256, upper bound: 180.4211776
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -174.8508941, upper bound: 180.2862085
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -174.1726636, upper bound: 179.7400062
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -174.1909384, upper bound: 179.6624718
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -118.3451233, 137.3436584, -144.5060425, 142.5423126, -260.8874207, 281.8496704
1: -97.5564499, 126.5701828, -119.1939545, 130.4303131, -227.9867096, 245.7641144
2: -136.5765839, 137.0777893, -166.8922729, 142.3275909, -278.9041748, 303.9700317
3: -76.5745544, 135.4180145, -82.8122025, 161.4513702, -238.0259247, 218.2301941
4: -148.6373901, 144.6610718, -181.0562744, 150.4014587, -299.0388489, 325.7173462

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -170.5663749, upper bound: 178.5476025
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -171.0417399, upper bound: 178.7830278
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -168.8096191, upper bound: 176.7922737
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -168.7582291, upper bound: 176.3955245
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -146.3338928, 155.7396545, -140.6025085, 139.6843872, -286.0182190, 296.3421631
1: -121.0181732, 143.3005981, -115.9742508, 127.7212601, -248.7393951, 259.2748413
2: -169.8362579, 154.3646698, -162.3865051, 139.2944336, -309.1306763, 316.7511597
3: -84.6786346, 167.1723022, -81.5001144, 157.3161316, -241.9947662, 248.6724243
4: -184.1595917, 162.9274750, -176.2256470, 147.1510620, -331.3106689, 339.1531372

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.4514035, upper bound: 186.7642730
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.1312636, upper bound: 187.3650590
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.6084517, upper bound: 186.9965663
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.7526292, upper bound: 186.9702479
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -150.8867340, 158.6611786, -95.6446609, 113.3904877, -264.2772217, 254.3058014
1: -124.7843018, 146.0037994, -79.2515564, 103.1769333, -227.9612427, 225.2553406
2: -175.1371307, 157.1483612, -111.1647339, 115.6497650, -290.7868347, 268.3130798
3: -86.2978363, 172.2207794, -69.2544556, 111.2451477, -197.5429688, 241.4752350
4: -189.8798828, 165.8626709, -120.8667679, 122.2402802, -312.1201782, 286.7294312

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.0125645, upper bound: 188.0517484
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.0646566, upper bound: 188.1345603
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.0646566, upper bound: 188.1345603
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -150.8867340, 158.6611786, -122.8770905, 132.3203735, -283.2070923, 281.5382690
1: -124.7843018, 146.0037994, -101.2287979, 121.3913574, -246.1756592, 247.2326050
2: -175.1371307, 157.1483612, -142.3907928, 133.3691559, -308.5062561, 299.5391541
3: -86.2978363, 172.2207794, -77.8224792, 140.9369202, -227.2347412, 250.0432587
4: -189.8798828, 165.8626709, -155.2873383, 140.4335175, -330.3134155, 321.1499939

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.0116105, upper bound: 188.0517484
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.0646566, upper bound: 188.1345631
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.0646566, upper bound: 188.1345631
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -141.8547516, 140.8447418, -177.3217163, 173.1061249, -314.9608765, 318.1664429
1: -116.9957733, 128.8614807, -146.2401733, 159.3694153, -276.3651733, 275.1016541
2: -163.8165283, 140.1550446, -204.8536377, 170.7242126, -334.5406799, 345.0086670
3: -81.7145462, 158.5599518, -94.0037155, 197.5484619, -279.2629700, 252.5636597
4: -177.7744446, 147.9893799, -222.1038513, 180.3532715, -358.1277161, 370.0932312

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -172.9173686, upper bound: 175.4226257
time: 0.59 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -174.2386387, upper bound: 176.3304260
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -170.8978894, upper bound: 175.6431994
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -170.4863975, upper bound: 170.8855274
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -81.8415833, 104.8546906, -182.5645752, 176.8494873, -258.6910706, 287.4192505
1: -68.0090408, 94.7844849, -150.5458374, 162.8777771, -230.8868103, 245.3303223
2: -95.3412018, 107.1040955, -210.8809204, 174.3390656, -269.6802673, 317.9850159
3: -64.9595795, 96.7800903, -95.6772766, 203.0880432, -268.0476074, 192.4573059
4: -103.6640930, 113.4794769, -228.6232605, 184.1927490, -287.8568420, 342.1026917

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.9861942, upper bound: 179.3318889
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2052321, upper bound: 179.4275135
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.2613831, upper bound: 177.6764674
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.9816028, upper bound: 173.3084914
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -98.1761932, 114.4870758, -182.9138031, 177.0833893, -275.2595520, 297.4008789
1: -81.2362823, 104.1438599, -150.8296814, 163.0941467, -244.3304138, 254.9734650
2: -113.9444351, 116.7703171, -211.2808533, 174.5656891, -288.5101013, 328.0511169
3: -69.7811661, 113.5294724, -95.7939606, 203.4659576, -273.2471313, 209.3234100
4: -123.9752121, 123.4040375, -229.0622559, 184.4309082, -308.4061279, 352.4663086

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.0981165, upper bound: 179.4275135
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.8974142, upper bound: 179.3318889
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.7143532, upper bound: 179.2424478
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -170.7523007, upper bound: 174.9647820
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.0297572, upper bound: 179.4653357
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -144.7281494, 142.6937256, -120.8671112, 139.6277618, -284.3558960, 263.5607910
1: -119.3738785, 130.5724030, -99.6032562, 128.4012909, -247.7751617, 230.1756592
2: -167.1474609, 142.4619598, -139.7168884, 139.6352539, -306.7826538, 282.1788330
3: -82.8798676, 161.6947021, -77.4247513, 138.7147980, -221.5946503, 239.1194458
4: -181.3374939, 150.5416412, -152.2470703, 147.2355194, -328.5729980, 302.7886963

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.4211776, upper bound: 174.9115256
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.2862085, upper bound: 174.8508941
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.7400062, upper bound: 174.1726636
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.6624718, upper bound: 174.1909384
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -144.5380096, 142.5588837, -118.3451233, 137.3436584, -281.8816528, 260.9039307
1: -119.2188416, 130.4456329, -97.5564499, 126.5701828, -245.7890167, 228.0020752
2: -166.9268036, 142.3426514, -136.5765839, 137.0777893, -304.0045776, 278.9192505
3: -82.8205185, 161.4807434, -76.5745544, 135.4180145, -218.2384949, 238.0552826
4: -181.0952606, 150.4170685, -148.6373901, 144.6610718, -325.7563477, 299.0544434

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.5476025, upper bound: 170.5663749
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.7830278, upper bound: 171.0417399
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.7922737, upper bound: 168.8096191
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3955245, upper bound: 168.7582291
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -140.6025085, 139.6843872, -146.3338928, 155.7396545, -296.3421631, 286.0182495
1: -115.9742508, 127.7212601, -121.0181732, 143.3005981, -259.2748413, 248.7394104
2: -162.3865051, 139.2944336, -169.8362579, 154.3646698, -316.7511292, 309.1306763
3: -81.5001144, 157.3161316, -84.6786346, 167.1723022, -248.6724243, 241.9947662
4: -176.2256470, 147.1510620, -184.1595917, 162.9274750, -339.1531372, 331.3106689

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.7642730, upper bound: 177.4514057
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3650563, upper bound: 178.1312635
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.9965663, upper bound: 177.6084517
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.9702486, upper bound: 177.7526301
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -95.6446609, 113.3904877, -150.8867340, 158.6611786, -254.3058167, 264.2772217
1: -79.2515564, 103.1769333, -124.7843018, 146.0037994, -225.2553406, 227.9612427
2: -111.1647339, 115.6497650, -175.1371307, 157.1483612, -268.3131104, 290.7868652
3: -69.2544556, 111.2451477, -86.2978363, 172.2207794, -241.4752197, 197.5429688
4: -120.8667679, 122.2402802, -189.8798828, 165.8626709, -286.7294312, 312.1201782

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.0679576, upper bound: 179.0120412
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_A2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.1491075, upper bound: 179.0646842
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.1491075, upper bound: 179.0646842
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -122.8770905, 132.3203735, -150.8867340, 158.6611786, -281.5382690, 283.2070923
1: -101.2287979, 121.3913574, -124.7843018, 146.0037994, -247.2325745, 246.1756592
2: -142.3907928, 133.3691559, -175.1371307, 157.1483612, -299.5391541, 308.5062256
3: -77.8224792, 140.9369202, -86.2978363, 172.2207794, -250.0432587, 227.2347412
4: -155.2873383, 140.4335175, -189.8798828, 165.8626709, -321.1499634, 330.3134155

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.0679576, upper bound: 179.0116778
time: 1.28 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_A2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.1491075, upper bound: 179.0646842
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.1491075, upper bound: 179.0646842
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -94.4513474, 118.2512894, -90.9153976, 112.4668808, -206.9182129, 209.1666870
1: -77.9498901, 107.6516876, -74.5072479, 101.4162598, -179.3661499, 182.1589355
2: -110.1855621, 119.3941422, -104.6127014, 114.8477783, -225.0333099, 224.0068359
3: -70.0200806, 114.1218262, -68.7031784, 103.4350052, -173.4550476, 182.8250122
4: -120.7585068, 125.8530807, -114.7270508, 121.5828171, -242.3413239, 240.5801392

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.6653737, upper bound: 173.0386154
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.1177013, upper bound: 171.1331668
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.4339472, upper bound: 172.9937909
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.1129215, upper bound: 170.6067927
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -170.6435996, upper bound: 168.7488492
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -144.1835785, 142.3243866, -91.4743271, 112.7514648, -256.9349976, 233.7987061
1: -118.9274597, 130.2256775, -74.9706650, 101.6991348, -220.6265869, 205.1963348
2: -166.5144348, 142.1378784, -105.2589417, 115.1343307, -281.6487732, 247.3968201
3: -82.7239838, 161.0882568, -68.8409653, 104.0475159, -186.7714996, 229.9292145
4: -180.6515503, 150.2081299, -115.4177475, 121.8774719, -302.5290222, 265.6258850

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.3063744, upper bound: 174.2160659
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.7203024, upper bound: 173.7785209
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A2_A2

### Relational analysis result of IS_A2_B2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.7914584, upper bound: 173.7797152
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -96.1266632, 119.3073578, -104.2965240, 118.7134628, -214.8401184, 223.6038361
1: -79.3496475, 108.6464310, -86.3277130, 108.2416611, -187.5913086, 194.9741516
2: -112.1503296, 120.2364273, -121.1016541, 120.4944992, -232.6448364, 241.3380737
3: -70.4191055, 115.9851532, -71.5731659, 120.4069977, -190.8260651, 187.5582886
4: -122.8487778, 126.7202454, -131.6491241, 127.2710495, -250.1198120, 258.3693848

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8738432, upper bound: 190.8835003
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.5620774, upper bound: 186.9020134
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.7818248, upper bound: 190.8642214
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -146.4110413, 143.7679901, -105.0934448, 119.2186661, -265.6296387, 248.8614044
1: -120.7598877, 131.5678406, -86.9803162, 108.7176056, -229.4774780, 218.5481567
2: -169.1005859, 143.3416443, -122.0204010, 120.9229431, -290.0235291, 265.3620300
3: -83.3419800, 163.5842438, -71.7856598, 121.2669067, -204.6088867, 235.3699036
4: -183.4533844, 151.4541473, -132.6455536, 127.7139664, -311.1673584, 284.0997009

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.9715486, upper bound: 191.0060334
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.8598699, upper bound: 191.0060334
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.8592631, upper bound: 190.9201313
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -91.4743271, 112.7514648, -130.9918976, 134.1106567, -225.5849304, 243.7433624
1: -74.9706650, 101.6991348, -108.0944977, 122.5496368, -197.5202942, 209.7936401
2: -105.2589417, 115.1343307, -151.3440247, 135.1260223, -240.3849335, 266.4783630
3: -68.8409653, 104.0475159, -79.1691895, 147.5771484, -216.4181213, 183.2167053
4: -115.4177475, 121.8774719, -164.4309692, 142.8990021, -258.3167419, 286.3084106

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -167.6168607, upper bound: 172.3179033
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_A1_B1_B1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -171.8573991, upper bound: 179.3392363
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -91.4743271, 112.7514648, -132.7845154, 133.6129761, -225.0872803, 245.5359802
1: -74.9706650, 101.6991348, -109.6106567, 121.9623337, -196.9329529, 211.3097839
2: -105.2589417, 115.1343307, -153.2735901, 134.5935059, -239.8524323, 268.4078979
3: -68.8409653, 104.0475159, -79.3009644, 147.4601135, -216.3010712, 183.3484802
4: -115.4177475, 121.8774719, -166.1767731, 142.4161682, -257.8338928, 288.0542297

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -170.4244583, upper bound: 177.5305638
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A2_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_A1_B1_B2_A1

### Relational analysis result of IS_A2_B2_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -169.1123886, upper bound: 176.3066253
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_A1_B1_B2_A2

### Relational analysis result of IS_A2_B2_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -169.1123886, upper bound: 177.5305625
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -76.8271179, 104.5473633, -107.1884460, 120.8012085, -197.6283264, 211.7358093
1: -63.1302948, 93.6720734, -88.6859131, 110.0723114, -173.2026062, 182.3579865
2: -88.4977875, 106.8460541, -124.4192505, 122.8460541, -211.3438416, 231.2652740
3: -64.9372559, 88.5096130, -72.6748505, 123.5625992, -188.4998474, 161.1844635
4: -96.9487228, 113.5803528, -135.2475128, 129.8513489, -226.8000793, 248.8278656

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -171.0334780, upper bound: 178.5444657
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -171.0621157, upper bound: 178.6005266
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -82.6301270, 107.1223373, -107.1884460, 120.8012085, -203.4313202, 214.3107605
1: -67.6292725, 96.0613403, -88.6859131, 110.0723114, -177.7015839, 184.7472534
2: -94.9032745, 109.5288544, -124.4192505, 122.8460541, -217.7493286, 233.9480896
3: -66.3763199, 93.4519348, -72.6748505, 123.5625992, -189.9389038, 166.1267853
4: -104.2620468, 116.0982208, -135.2475128, 129.8513489, -234.1134033, 251.3457336

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -171.9389121, upper bound: 179.3414778
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -171.9521752, upper bound: 179.4081221
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -105.0934448, 119.2186661, -120.0220337, 128.6931763, -233.7866211, 239.2406921
1: -86.9803162, 108.7176056, -99.2100220, 117.5533295, -204.5336456, 207.9276123
2: -122.0204010, 120.9229431, -138.8724823, 129.1933289, -251.2137299, 259.7954102
3: -71.7856598, 121.2669067, -75.8478699, 136.0002441, -207.7859039, 197.1147766
4: -132.6455536, 127.7139664, -150.6780090, 136.6561279, -269.3016968, 278.3919678

Time for backsubstitution: 2.22 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=279.9653625488281
rel_dist={0: [-191.43388219474815, 191.4338821947481]}

## Binary search (step 1) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.3499641, upper bound: 183.6521416
time: 0.62 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1345495, upper bound: 191.1345518
time: 0.57 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.40 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.40
Output dim: 0, lower bound: -179.3499641, upper bound: 183.6521416
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.40
Output dim: 0, lower bound: -191.1345495, upper bound: 191.1345518

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -165.7317810, 169.9891968, -115.4142151, 128.1499329, -293.8816833, 285.4034119
1: -136.9342194, 156.5604401, -95.3663254, 116.9463272, -253.8805237, 251.9267273
2: -192.2410583, 168.2551727, -133.9529266, 129.7927856, -322.0337524, 302.2080688
3: -91.3198547, 188.0007629, -75.9782944, 132.4578400, -223.7776947, 263.9790649
4: -208.4816437, 177.5048828, -145.6999359, 137.1465759, -345.6282043, 323.2047729

Time for backsubstitution: 2.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.9588462, upper bound: 183.4782680
time: 0.67 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.9495230, upper bound: 183.4926866
time: 0.69 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -126.3109131, 134.7957306, -138.0393982, 141.9259644, -268.2368774, 272.8351440
1: -104.3817749, 123.2117767, -113.9848785, 129.8824921, -234.2642670, 237.1966400
2: -146.5827942, 135.2793121, -160.1121521, 141.6186523, -288.2014465, 295.3914795
3: -78.7137909, 144.2006378, -81.8182144, 156.6791382, -235.3929138, 226.0188599
4: -159.2684784, 142.8177032, -173.9111633, 149.4060059, -308.6744995, 316.7288818

Time for backsubstitution: 2.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.6521416, upper bound: 179.3499641
time: 0.71 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.6521416, upper bound: 191.1345518
time: 0.63 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.37 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.37
Output dim: 0, lower bound: -178.9588462, upper bound: 183.4782680
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.37
Output dim: 0, lower bound: -178.9495230, upper bound: 183.4926866
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.37
Output dim: 0, lower bound: -183.6521416, upper bound: 179.3499641
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.37
Output dim: 0, lower bound: -183.6521416, upper bound: 191.1345518

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -160.0070038, 166.0839386, -133.8944244, 136.3115234, -296.3185425, 299.9783325
1: -132.3008881, 152.9327850, -110.4975281, 124.5818787, -256.8827209, 263.4302979
2: -185.7236938, 164.4482574, -154.6473236, 137.0007324, -322.7244263, 319.0954895
3: -89.4389801, 181.9106903, -80.0173264, 150.3701782, -239.8090973, 261.9280090
4: -201.2560120, 173.5649872, -167.8708649, 144.8641815, -346.1201782, 341.4357910

Time for backsubstitution: 2.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.3339475, upper bound: 177.5463319
time: 0.70 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.3339475, upper bound: 183.4782680
time: 0.76 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -162.2162476, 167.2798004, -98.3869934, 116.4440460, -278.6602783, 265.6668091
1: -134.0574341, 154.0412903, -81.3735199, 105.4789352, -239.5363617, 235.4148102
2: -188.1860352, 165.6090393, -114.1667862, 118.4898605, -306.6759033, 279.7758179
3: -90.1217651, 184.2256775, -70.5018997, 114.1076965, -204.2294617, 254.7275238
4: -204.0666809, 174.7288208, -124.3289185, 125.3610687, -329.4277344, 299.0577393

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2475539, upper bound: 177.2475535
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2475539, upper bound: 183.4926866
time: 0.60 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -126.3109131, 134.7957306, -165.7317810, 169.9891968, -296.3001099, 300.5275269
1: -104.3817749, 123.2117767, -136.9342194, 156.5604401, -260.9421997, 260.1459961
2: -146.5827942, 135.2793121, -192.2410583, 168.2551727, -314.8379517, 327.5202942
3: -78.7137909, 144.2006378, -91.3198547, 188.0007629, -266.7145081, 235.5204926
4: -159.2684784, 142.8177032, -208.4816437, 177.5048828, -336.7733765, 351.2993469

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.5688741, upper bound: 179.2490221
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.4926836, upper bound: 178.9495222
time: 0.65 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -126.3109131, 134.7957306, -126.3109131, 134.7957306, -261.1066284, 261.1066284
1: -104.3817749, 123.2117767, -104.3817749, 123.2117767, -227.5935364, 227.5935211
2: -146.5827942, 135.2793121, -146.5827942, 135.2793121, -281.8620911, 281.8620911
3: -78.7137909, 144.2006378, -78.7137909, 144.2006378, -222.9144287, 222.9144287
4: -159.2684784, 142.8177032, -159.2684784, 142.8177032, -302.0861816, 302.0861816

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.4782680, upper bound: 191.0597565
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.4926866, upper bound: 191.0593396
time: 0.67 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.21 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 4.21
Output dim: 0, lower bound: -177.3339475, upper bound: 177.5463319
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.21
Output dim: 0, lower bound: -177.3339475, upper bound: 183.4782680
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.21
Output dim: 0, lower bound: -177.2475539, upper bound: 177.2475535
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.21
Output dim: 0, lower bound: -177.2475539, upper bound: 183.4926866
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 4.21
Output dim: 0, lower bound: -183.5688741, upper bound: 179.2490221
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.21
Output dim: 0, lower bound: -183.4926836, upper bound: 178.9495222
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.21
Output dim: 0, lower bound: -183.4782680, upper bound: 191.0597565
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.21
Output dim: 0, lower bound: -183.4926866, upper bound: 191.0593396

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -160.0070038, 166.0839386, -180.9865112, 176.3051453, -336.3121338, 347.0704346
1: -132.3008881, 152.9327850, -149.3313446, 162.4405060, -294.7413940, 302.2641296
2: -185.7236938, 164.4482574, -209.1234741, 173.9082642, -359.6319580, 373.5716553
3: -89.4389801, 181.9106903, -95.2905579, 202.0634460, -291.5024109, 277.2012329
4: -201.2560120, 173.5649872, -226.7189331, 183.6530762, -384.9090881, 400.2838745

Time for backsubstitution: 2.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.3339475, upper bound: 177.5463319
time: 0.77 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.3339475, upper bound: 177.5463319
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -160.0070038, 166.0839386, -144.1305847, 142.7381897, -302.7451477, 310.2145081
1: -132.3008881, 152.9327850, -118.9544830, 130.6557159, -262.9565735, 271.8872681
2: -185.7236938, 164.4482574, -166.5420380, 142.3909149, -328.1145935, 330.9902954
3: -89.4389801, 181.9106903, -82.7372742, 161.5991821, -251.0381622, 264.6479492
4: -201.2560120, 173.5649872, -180.6805878, 150.4187775, -351.6747437, 354.2455444

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.3339475, upper bound: 183.4773905
time: 0.72 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.3339475, upper bound: 183.4782680
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -162.2162476, 167.2798004, -148.8708801, 158.1273041, -320.3435364, 316.1506348
1: -134.0574341, 154.0412903, -123.2851181, 145.4833221, -279.5407104, 277.3264160
2: -188.1860352, 165.6090393, -172.9945068, 156.6401520, -344.8261719, 338.6035461
3: -90.1217651, 184.2256775, -85.9972076, 170.7021332, -260.8239136, 270.2228394
4: -204.0666809, 174.7288208, -187.4190979, 165.2965240, -369.3631592, 362.1479187

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.5582598, upper bound: 176.4758593
time: 0.56 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3630997, upper bound: 176.3630985
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -162.2162476, 167.2798004, -109.0595627, 121.9669266, -284.1831665, 276.3393555
1: -134.0574341, 154.0412903, -90.2522430, 111.2593994, -245.3168335, 244.2935181
2: -188.1860352, 165.6090393, -126.6313934, 123.8755035, -312.0614929, 292.2403870
3: -90.1217651, 184.2256775, -73.1986008, 125.7491608, -215.8709259, 257.4242554
4: -204.0666809, 174.7288208, -137.6091766, 130.9158630, -334.9825439, 312.3379822

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2352643, upper bound: 183.4917902
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2352643, upper bound: 183.4922852
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -120.5661392, 130.9839783, -182.9138031, 177.0833893, -297.6495361, 313.8977661
1: -99.7544785, 119.6763535, -150.8296814, 163.0941467, -262.8486328, 270.5060425
2: -140.0446320, 131.8153229, -211.2808533, 174.5656891, -314.6102905, 343.0961914
3: -76.9492798, 138.0948639, -95.7939606, 203.4659576, -280.4152222, 233.8888245
4: -152.0723572, 139.2584839, -229.0622559, 184.4309082, -336.5032349, 368.3207397

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.4768546, upper bound: 178.9372229
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.4768546, upper bound: 178.9372229
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -122.2428284, 131.6734314, -150.8867340, 158.6611786, -280.9039917, 282.5601501
1: -101.0460510, 120.3123322, -124.7843018, 146.0037994, -247.0498505, 245.0966339
2: -141.8740234, 132.5205383, -175.1371307, 157.1483612, -299.0223999, 307.6576233
3: -77.3885574, 139.7854614, -86.2978363, 172.2207794, -249.6093292, 226.0832977
4: -154.1522522, 139.9355011, -189.8798828, 165.8626709, -320.0149231, 329.8153687

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.4777540, upper bound: 178.9495226
time: 0.80 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.4777540, upper bound: 178.9495226
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -146.9546051, 144.1370697, -120.5661392, 130.9839783, -277.9385986, 264.7032166
1: -121.2054749, 131.9143066, -99.7544785, 119.6763535, -240.8818207, 231.6687622
2: -169.7323761, 143.6659851, -140.0446320, 131.8153229, -301.5476990, 283.7106323
3: -83.4981003, 164.1896667, -76.9492798, 138.0948639, -221.5929565, 241.1389465
4: -184.1379089, 151.7879486, -152.0723572, 139.2584839, -323.3963623, 303.8602905

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0334011, upper bound: 191.0597413
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0334846, upper bound: 191.0334846
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0334846, upper bound: 191.0592473
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -109.4849014, 122.1597290, -122.2428284, 131.6734314, -241.1583252, 244.4025421
1: -90.5823822, 111.4376678, -101.0460510, 120.3123322, -210.8946991, 212.4837189
2: -127.0952301, 124.0488815, -141.8740234, 132.5205383, -259.6157837, 265.9229126
3: -73.2953796, 126.1313477, -77.3885574, 139.7854614, -213.0808411, 203.5198975
4: -138.1383820, 131.0992584, -154.1522522, 139.9355011, -278.0738831, 285.2515259

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0591960, upper bound: 191.0532659
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0535180, upper bound: 191.0593295
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0591140, upper bound: 191.0334846
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0591140, upper bound: 191.0593372
time: 0.75 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 6.81 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 0, lower bound: -177.3339475, upper bound: 177.5463319
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 0, lower bound: -177.3339475, upper bound: 177.5463319
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 0, lower bound: -177.3339475, upper bound: 183.4773905
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 0, lower bound: -177.3339475, upper bound: 183.4782680
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 0, lower bound: -176.5582598, upper bound: 176.4758593
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 0, lower bound: -176.3630997, upper bound: 176.3630985
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 0, lower bound: -177.2352643, upper bound: 183.4917902
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 0, lower bound: -177.2352643, upper bound: 183.4922852
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 0, lower bound: -183.4768546, upper bound: 178.9372229
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 0, lower bound: -183.4768546, upper bound: 178.9372229
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 0, lower bound: -183.4777540, upper bound: 178.9495226
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 0, lower bound: -183.4777540, upper bound: 178.9495226
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 0, lower bound: -191.0334846, upper bound: 191.0334846
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 0, lower bound: -191.0334846, upper bound: 191.0592473
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 0, lower bound: -191.0591140, upper bound: 191.0334846
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 0, lower bound: -191.0591140, upper bound: 191.0593372

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -182.9076538, 176.9971466, -180.9865112, 176.3051453, -359.2127991, 357.9836426
1: -150.8231964, 163.0276184, -149.3313446, 162.4405060, -313.2637024, 312.3589478
2: -211.2644501, 174.4967804, -209.1234741, 173.9082642, -385.1727295, 383.6202393
3: -95.7824173, 203.4107056, -95.2905579, 202.0634460, -297.8457642, 298.7012634
4: -229.0425720, 184.3425751, -226.7189331, 183.6530762, -412.6955872, 411.0615234

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3387692, upper bound: 176.0488047
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.8481590, upper bound: 175.9048567
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -149.8873901, 157.9709930, -180.9865112, 176.3051453, -326.1925354, 338.9575195
1: -123.9680481, 145.3655548, -149.3313446, 162.4405060, -286.4085693, 294.6968689
2: -173.9833679, 156.4939575, -209.1234741, 173.9082642, -347.8916016, 365.6174011
3: -85.9076080, 171.1465759, -95.2905579, 202.0634460, -287.9710083, 266.4371338
4: -188.5898285, 165.1896057, -226.7189331, 183.6530762, -372.2429199, 391.9085388

Time for backsubstitution: 2.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -173.0823268, upper bound: 175.8298953
time: 0.67 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9991572, upper bound: 177.2383178
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -182.9076538, 176.9971466, -144.1305847, 142.7381897, -325.6458435, 321.1277161
1: -150.8231964, 163.0276184, -118.9544830, 130.6557159, -281.4789124, 281.9820862
2: -211.2644501, 174.4967804, -166.5420380, 142.3909149, -353.6553345, 341.0388184
3: -95.7824173, 203.4107056, -82.7372742, 161.5991821, -257.3815613, 286.1479797
4: -229.0425720, 184.3425751, -180.6805878, 150.4187775, -379.4611816, 365.0231628

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.2572901, upper bound: 181.6682972
time: 0.73 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -174.7181758, upper bound: 177.1619144
time: 0.70 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -169.9257327, upper bound: 173.7603666
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -149.8873901, 157.9709930, -144.1305847, 142.7381897, -292.6255798, 302.1015625
1: -123.9680481, 145.3655548, -118.9544830, 130.6557159, -254.6237640, 264.3200378
2: -173.9833679, 156.4939575, -166.5420380, 142.3909149, -316.3742065, 323.0360107
3: -85.9076080, 171.1465759, -82.7372742, 161.5991821, -247.5067902, 253.8838501
4: -188.5898285, 165.1896057, -180.6805878, 150.4187775, -339.0085144, 345.8701782

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -173.7719494, upper bound: 178.8586481
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.6989890, upper bound: 183.2327565
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -144.3281250, 155.6313019, -143.8832397, 154.6725769, -299.0007019, 299.5144653
1: -119.6110535, 143.0676880, -119.2611465, 142.2241974, -261.8352661, 262.3287964
2: -167.8098755, 154.2328949, -167.3558655, 153.2641144, -321.0739746, 321.5887451
3: -84.7343521, 165.2545624, -84.5588989, 165.3638611, -250.0981598, 249.8134613
4: -181.6195984, 162.9631500, -181.0666504, 161.8355255, -343.4550781, 344.0297852

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.5582598, upper bound: 176.4758593
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.5292007, upper bound: 176.4660804
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -148.9003601, 158.1014862, -146.3477020, 156.2618408, -305.1621399, 304.4491577
1: -123.1389236, 145.3568878, -121.1911392, 143.7122803, -266.8511963, 266.5480347
2: -172.7638397, 156.7149811, -170.0382385, 154.8197632, -327.5835571, 326.7532043
3: -86.1955795, 168.8132172, -85.2215042, 167.6291199, -253.8246460, 254.0347290
4: -187.2846985, 165.3796692, -184.2139282, 163.3877869, -350.6724854, 349.5935974

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.0996686, upper bound: 176.3228313
time: 0.78 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3630997, upper bound: 176.3630985
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -182.9076538, 176.9971466, -109.0595627, 121.9669266, -304.8745728, 286.0567017
1: -150.8231964, 163.0276184, -90.2522430, 111.2593994, -262.0825500, 253.2798462
2: -211.2644501, 174.4967804, -126.6313934, 123.8755035, -335.1399231, 301.1281128
3: -95.7824173, 203.4107056, -73.1986008, 125.7491608, -221.5315552, 276.6093140
4: -229.0425720, 184.3425751, -137.6091766, 130.9158630, -359.9583740, 321.9517517

Time for backsubstitution: 3.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.7932053, upper bound: 171.7798601
time: 0.97 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.6835845, upper bound: 183.2384627
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -150.8867340, 158.6611786, -109.0595627, 121.9669266, -272.8536682, 267.7207336
1: -124.7843018, 146.0037994, -90.2522430, 111.2593994, -236.0437012, 236.2559967
2: -175.1371307, 157.1483612, -126.6313934, 123.8755035, -299.0126038, 283.7797546
3: -86.2978363, 172.2207794, -73.1986008, 125.7491608, -212.0469818, 245.4193726
4: -189.8798828, 165.8626709, -137.6091766, 130.9158630, -320.7957458, 303.4718323

Time for backsubstitution: 3.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.7932053, upper bound: 171.9546274
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.6835845, upper bound: 183.2359957
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -145.2131958, 143.3489685, -182.9138031, 177.0833893, -322.2965698, 326.2627563
1: -119.8059006, 131.2140198, -150.8296814, 163.0941467, -282.9000549, 282.0437012
2: -167.7163086, 142.9577484, -211.2808533, 174.5656891, -342.2819824, 354.2385864
3: -83.0653000, 162.6521912, -95.7939606, 203.4659576, -286.5312500, 258.4461365
4: -181.9947968, 151.0111847, -229.0622559, 184.4309082, -366.4255981, 380.0734253

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.8253095, upper bound: 178.7948920
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.6247518, upper bound: 173.2383660
time: 0.58 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.5562833, upper bound: 171.7140391
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -109.4849014, 122.1597290, -182.9138031, 177.0833893, -286.5682983, 305.0735474
1: -90.5823822, 111.4376678, -150.8296814, 163.0941467, -253.6765289, 262.2673035
2: -127.0952301, 124.0488815, -211.2808533, 174.5656891, -301.6609192, 335.3297424
3: -73.2953796, 126.1313477, -95.7939606, 203.4659576, -276.7613525, 221.9253082
4: -138.1383820, 131.0992584, -229.0622559, 184.4309082, -322.5691833, 360.1614990

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.2669906, upper bound: 175.3563103
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.3221738, upper bound: 179.0247094
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -146.9546051, 144.1370697, -150.8867340, 158.6611786, -305.6157837, 295.0237732
1: -121.2054749, 131.9143066, -124.7843018, 146.0037994, -267.2092590, 256.6986084
2: -169.7323761, 143.6659851, -175.1371307, 157.1483612, -326.8807068, 318.8030701
3: -83.4981003, 164.1896667, -86.2978363, 172.2207794, -255.7188721, 250.4874878
4: -184.1379089, 151.7879486, -189.8798828, 165.8626709, -350.0005493, 341.6678467

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.4220406, upper bound: 174.0339609
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.2278058, upper bound: 178.6942158
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -109.4849014, 122.1597290, -150.8867340, 158.6611786, -268.1460876, 273.0464172
1: -90.5823822, 111.4376678, -124.7843018, 146.0037994, -236.5861816, 236.2219543
2: -127.0952301, 124.0488815, -175.1371307, 157.1483612, -284.2435913, 299.1860046
3: -73.2953796, 126.1313477, -86.2978363, 172.2207794, -245.5161591, 212.4291840
4: -138.1383820, 131.0992584, -189.8798828, 165.8626709, -304.0009766, 320.9791260

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.1549781, upper bound: 175.3884120
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.2278047, upper bound: 178.6939569
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -146.9546051, 144.1370697, -145.2131958, 143.3489685, -290.3035889, 289.3502197
1: -121.2054749, 131.9143066, -119.8059006, 131.2140198, -252.4194946, 251.7201996
2: -169.7323761, 143.6659851, -167.7163086, 142.9577484, -312.6900940, 311.3822632
3: -83.4981003, 164.1896667, -83.0653000, 162.6521912, -246.1502991, 247.2549744
4: -184.1379089, 151.7879486, -181.9947968, 151.0111847, -335.1491089, 333.7827148

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.4436976, upper bound: 177.9222641
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0042211, upper bound: 176.0622054
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -146.9546051, 144.1370697, -109.4849014, 122.1597290, -269.1143188, 253.6219788
1: -121.2054749, 131.9143066, -90.5823822, 111.4376678, -232.6431274, 222.4966736
2: -169.7323761, 143.6659851, -127.0952301, 124.0488815, -293.7812500, 270.7612305
3: -83.4981003, 164.1896667, -73.2953796, 126.1313477, -209.6294403, 237.4850464
4: -184.1379089, 151.7879486, -138.1383820, 131.0992584, -315.2371826, 289.9262695

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.3445062, upper bound: 172.8753682
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.9716494, upper bound: 191.0013349
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -109.4849014, 122.1597290, -146.9546051, 144.1370697, -253.6219788, 269.1143188
1: -90.5823822, 111.4376678, -121.2054749, 131.9143066, -222.4966888, 232.6431122
2: -127.0952301, 124.0488815, -169.7323761, 143.6659851, -270.7612305, 293.7812500
3: -73.2953796, 126.1313477, -83.4981003, 164.1896667, -237.4850311, 209.6294403
4: -138.1383820, 131.0992584, -184.1379089, 151.7879486, -289.9262695, 315.2371826

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0591071, upper bound: 191.0333963
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.7798601, upper bound: 175.3547269
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0007966, upper bound: 190.9725121
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -109.4849014, 122.1597290, -109.4849014, 122.1597290, -231.6446228, 231.6446228
1: -90.5823822, 111.4376678, -90.5823822, 111.4376678, -202.0200043, 202.0200043
2: -127.0952301, 124.0488815, -127.0952301, 124.0488815, -251.1441040, 251.1441040
3: -73.2953796, 126.1313477, -73.2953796, 126.1313477, -199.4267120, 199.4267273
4: -138.1383820, 131.0992584, -138.1383820, 131.0992584, -269.2376404, 269.2376404

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.7798625, upper bound: 175.8078490
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0007974, upper bound: 191.0009643
time: 0.65 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.84 seconds
IS_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 0, lower bound: -176.3387692, upper bound: 176.0488047
IS_A1_B1_B1_A1_A2, status: Status.VERIFIED, split count: 5, time: 4.84
Output dim: 0, lower bound: -175.8481590, upper bound: 175.9048567
IS_A1_B1_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 4.84
Output dim: 0, lower bound: -173.0823268, upper bound: 175.8298953
IS_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 0, lower bound: -176.9991572, upper bound: 177.2383178
IS_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 0, lower bound: -174.7181758, upper bound: 177.1619144
IS_A1_B1_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 4.84
Output dim: 0, lower bound: -169.9257327, upper bound: 173.7603666
IS_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 0, lower bound: -173.7719494, upper bound: 178.8586481
IS_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 0, lower bound: -178.6989890, upper bound: 183.2327565
IS_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 0, lower bound: -176.5582598, upper bound: 176.4758593
IS_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 0, lower bound: -176.5292007, upper bound: 176.4660804
IS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 0, lower bound: -176.0996686, upper bound: 176.3228313
IS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 0, lower bound: -176.3630997, upper bound: 176.3630985
IS_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 4.84
Output dim: 0, lower bound: -174.7932053, upper bound: 171.7798601
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 0, lower bound: -178.6835845, upper bound: 183.2384627
IS_A1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 4.84
Output dim: 0, lower bound: -174.7932053, upper bound: 171.9546274
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 0, lower bound: -178.6835845, upper bound: 183.2359957
IS_A2_B1_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 4.84
Output dim: 0, lower bound: -175.6247518, upper bound: 173.2383660
IS_A2_B1_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 4.84
Output dim: 0, lower bound: -174.5562833, upper bound: 171.7140391
IS_A2_B1_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 4.84
Output dim: 0, lower bound: -171.2669906, upper bound: 175.3563103
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 0, lower bound: -183.3221738, upper bound: 179.0247094
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 0, lower bound: -178.4220406, upper bound: 174.0339609
IS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 0, lower bound: -183.2278058, upper bound: 178.6942158
IS_A2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 4.84
Output dim: 0, lower bound: -171.1549781, upper bound: 175.3884120
IS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 0, lower bound: -183.2278047, upper bound: 178.6939569
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 0, lower bound: -183.4436976, upper bound: 177.9222641
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.84
Output dim: 0, lower bound: -176.0042211, upper bound: 176.0622054
IS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 0, lower bound: -178.3445062, upper bound: 172.8753682
IS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 0, lower bound: -190.9716494, upper bound: 191.0013349
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.84
Output dim: 0, lower bound: -171.7798601, upper bound: 175.3547269
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 0, lower bound: -191.0007966, upper bound: 190.9725121
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.84
Output dim: 0, lower bound: -171.7798625, upper bound: 175.8078490
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 0, lower bound: -191.0007974, upper bound: 191.0009643

## BFS IS instance: IS_A1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -177.3173370, 173.0451965, -180.9865112, 176.3051453, -353.6224365, 354.0317078
1: -146.2355804, 159.3224487, -149.3313446, 162.4405060, -308.6760864, 308.6537781
2: -204.8420105, 170.6755066, -209.1234741, 173.9082642, -378.7502747, 379.7989807
3: -93.9955902, 197.5093689, -95.2905579, 202.0634460, -296.0590210, 292.7999268
4: -222.0899048, 180.2908630, -226.7189331, 183.6530762, -405.7429504, 407.0097656

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9325601, upper bound: 175.9325601
time: 0.69 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9325601, upper bound: 175.9325601
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -145.4437866, 155.1196899, -180.9865112, 176.3051453, -321.7489014, 336.1062012
1: -120.2910233, 142.7280731, -149.3313446, 162.4405060, -282.7315369, 292.0593872
2: -168.8097839, 153.7775879, -209.1234741, 173.9082642, -342.7180481, 362.9010315
3: -84.3281860, 166.2122345, -95.2905579, 202.0634460, -286.3916321, 261.5028076
4: -183.0110016, 162.3237000, -226.7189331, 183.6530762, -366.6640625, 389.0426331

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9818868, upper bound: 177.1708085
time: 0.65 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.1499465, upper bound: 176.2530755
time: 0.70 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.8536255, upper bound: 176.4882298
time: 0.68 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.7003913, upper bound: 175.6980359
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -169.1473541, 166.2080994, -143.9395599, 142.5865479, -311.7338867, 310.1476440
1: -139.5300598, 152.7494812, -118.7973938, 130.5121918, -270.0421448, 271.5468750
2: -195.3223114, 164.0295563, -166.3203735, 142.2481995, -337.5704346, 330.3498535
3: -91.0397263, 187.8306580, -82.6720886, 161.3802185, -252.4199524, 270.5027466
4: -211.7273102, 173.3522797, -180.4400024, 150.2694092, -361.9966736, 353.7922668

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_A1_A1_A1

### Relational analysis result of IS_A1_B1_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -166.7684289, upper bound: 168.6136293
time: 0.72 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -173.8339612, upper bound: 176.1229116
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.9810240, upper bound: 169.1308873
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -132.2132874, 148.6410675, -131.3715210, 134.3198547, -266.5331421, 280.0125427
1: -109.0739136, 137.1258698, -108.4150620, 122.8092346, -231.8831177, 245.5409241
2: -153.0550537, 148.4509735, -151.6669006, 135.4158478, -288.4708862, 300.1178589
3: -81.0738907, 151.7763062, -79.2007294, 147.0768738, -228.1507263, 230.9770203
4: -166.4611969, 156.4946899, -164.5923462, 143.2014771, -309.6626587, 321.0870361

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -174.0339609, upper bound: 178.8586475
time: 0.75 seconds

## Relational analysis of IS_A1_B1_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -168.2598253, upper bound: 170.3459127
time: 0.74 seconds

## Relational analysis of IS_A1_B1_B2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -172.4279244, upper bound: 178.2434292
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -172.5392970, upper bound: 175.0792237
time: 0.73 seconds

## Relational analysis of IS_A1_B1_B2_A2_A1_A2

### Relational analysis result of IS_A1_B1_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -169.6473842, upper bound: 173.4643507
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -145.4437866, 155.1196899, -144.1305847, 142.7381897, -288.1818848, 299.2502136
1: -120.2910233, 142.7280731, -118.9544830, 130.6557159, -250.9467468, 261.6825562
2: -168.8097839, 153.7775879, -166.5420380, 142.3909149, -311.2006836, 320.3196411
3: -84.3281860, 166.2122345, -82.7372742, 161.5991821, -245.9273682, 248.9495087
4: -183.0110016, 162.3237000, -180.6805878, 150.4187775, -333.4296570, 343.0042725

Time for backsubstitution: 2.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.7054138, upper bound: 183.2313428
time: 0.66 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_A2_A2_A1

### Relational analysis result of IS_A1_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.0171832, upper bound: 181.5145253
time: 0.74 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2_A2

### Relational analysis result of IS_A1_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.5617945, upper bound: 181.4154661
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -134.5135040, 149.5867004, -143.0968933, 154.1893005, -288.7027283, 292.6835938
1: -111.5995026, 137.4353180, -118.6163406, 141.7739563, -253.3734589, 256.0516663
2: -156.6116943, 148.4280853, -166.4559174, 152.7984772, -309.4101562, 314.8839722
3: -81.5965118, 154.7332458, -84.3095016, 164.5245209, -246.1210327, 239.0427551
4: -169.4080505, 156.8603821, -180.0887299, 161.3464355, -330.7544861, 336.9490967

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.4909771, upper bound: 176.2230518
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.4909771, upper bound: 176.4660804
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -168.1466064, 173.6803131, -136.4254608, 149.9073944, -318.0540161, 310.1057129
1: -138.6561737, 160.4846039, -113.1143799, 137.8236694, -276.4798584, 273.5989685
2: -194.8035889, 173.0832062, -158.7315216, 148.7917175, -343.5953064, 331.8146362
3: -92.7270432, 191.0919342, -82.1583939, 157.1592102, -249.8862152, 273.2503357
4: -211.6338806, 182.2114868, -171.7245178, 157.1845398, -368.8184204, 353.9360046

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.4909771, upper bound: 176.2230518
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.4909771, upper bound: 176.4660804
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -138.8819580, 152.1155396, -145.5466461, 155.7678375, -294.6497803, 297.6621704
1: -115.0018082, 139.7933044, -120.5357819, 143.2522736, -258.2540894, 260.3291016
2: -161.3812408, 150.9663696, -169.1232758, 154.3442993, -315.7255249, 320.0896606
3: -83.0931854, 158.2641449, -84.9651718, 166.7726288, -249.8657990, 243.2292938
4: -174.8187408, 159.3495178, -183.2173004, 162.8883209, -337.7070007, 342.5667419

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0620677, upper bound: 176.0620666
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.0620677, upper bound: 176.3228313
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -168.0211487, 173.2003021, -138.8969574, 151.5057831, -319.5269165, 312.0972595
1: -138.5490112, 159.9856415, -115.0714111, 139.3151398, -277.8641357, 275.0570374
2: -194.7471924, 172.6358185, -161.4555054, 150.3537140, -345.1008911, 334.0913086
3: -92.8779449, 190.8125153, -82.8180618, 159.4476013, -252.3255463, 273.6304932
4: -211.6457520, 181.7321167, -174.8841705, 158.7464294, -370.3921204, 356.6162720

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3228325, upper bound: 176.0996675
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3228325, upper bound: 176.3630985
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -182.9076538, 176.9971466, -105.0257416, 119.1997681, -302.1074219, 282.0228882
1: -150.8231964, 163.0276184, -86.9293213, 108.6992035, -259.5223694, 249.9569092
2: -211.2644501, 174.4967804, -121.9474335, 120.9054489, -332.1698914, 296.4442139
3: -95.7824173, 203.4107056, -71.7761078, 121.2145233, -216.9969330, 275.1868286
4: -229.0425720, 184.3425751, -132.5622253, 127.6949005, -356.7373962, 316.9047852

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.9769150, upper bound: 183.3103504
time: 0.56 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.8084666, upper bound: 183.2321584
time: 0.75 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.0247094, upper bound: 183.3221738
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -150.8867340, 158.6611786, -105.0257416, 119.1997681, -270.0864868, 263.6869202
1: -124.7843018, 146.0037994, -86.9293213, 108.6992035, -233.4835052, 232.9331055
2: -175.1371307, 157.1483612, -121.9474335, 120.9054489, -296.0425415, 279.0957947
3: -86.2978363, 172.2207794, -71.7761078, 121.2145233, -207.5123596, 243.9968719
4: -189.8798828, 165.8626709, -132.5622253, 127.6949005, -317.5747681, 298.4248962

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.6941563, upper bound: 183.2355908
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.6939077, upper bound: 183.2355940
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -105.0934448, 119.2186661, -182.9138031, 177.0833893, -282.1768188, 302.1324768
1: -86.9803162, 108.7176056, -150.8296814, 163.0941467, -250.0744629, 259.5472717
2: -122.0204010, 120.9229431, -211.2808533, 174.5656891, -296.5860901, 332.2037964
3: -71.7856598, 121.2669067, -95.7939606, 203.4659576, -275.2516174, 217.0608673
4: -132.6455536, 127.7139664, -229.0622559, 184.4309082, -317.0764465, 356.7762146

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.3103472, upper bound: 178.9769144
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.2321584, upper bound: 178.8084666
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.3221738, upper bound: 179.0247094
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -133.4149170, 135.3246155, -132.5038147, 148.8401642, -282.2550659, 267.8283997
1: -110.0467529, 123.7133789, -109.3140106, 137.3092957, -247.3560028, 233.0273895
2: -153.9834900, 136.3329620, -153.3966522, 148.6413116, -302.6247559, 289.7296143
3: -79.7415695, 148.9605865, -81.1663208, 152.0792236, -231.8207855, 230.1269073
4: -167.0974121, 144.1869965, -166.8361816, 156.6905060, -323.7878723, 311.0231018

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.8586481, upper bound: 174.0339609
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -170.3459127, upper bound: 168.2598253
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.2434300, upper bound: 172.4279272
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_B2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.0792237, upper bound: 172.5392970
time: 0.80 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -173.4643507, upper bound: 169.6473842
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -146.9546051, 144.1370697, -146.3338928, 155.7396545, -302.6942749, 290.4709473
1: -121.2054749, 131.9143066, -121.0181732, 143.3005981, -264.5060730, 252.9324799
2: -169.7323761, 143.6659851, -169.8362579, 154.3646698, -324.0970154, 313.5021667
3: -83.4981003, 164.1896667, -84.6786346, 167.1723022, -250.6703796, 248.8683014
4: -184.1379089, 151.7879486, -184.1595917, 162.9274750, -347.0653687, 335.9475403

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.2313428, upper bound: 178.7054138
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.5145253, upper bound: 178.0171832
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.4154627, upper bound: 177.5617945
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -105.0934448, 119.2186661, -150.8867340, 158.6611786, -263.7546082, 270.1054077
1: -86.9803162, 108.7176056, -124.7843018, 146.0037994, -232.9841156, 233.5019073
2: -122.0204010, 120.9229431, -175.1371307, 157.1483612, -279.1687622, 296.0600281
3: -71.7856598, 121.2669067, -86.2978363, 172.2207794, -244.0064392, 207.5647430
4: -132.6455536, 127.7139664, -189.8798828, 165.8626709, -298.5082397, 317.5938416

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_A2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.2370949, upper bound: 178.6935603
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.2370949, upper bound: 178.6935068
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -140.6025085, 139.6843872, -145.2131958, 143.3489685, -283.9514465, 284.8974915
1: -115.9742508, 127.7212601, -119.8059006, 131.2140198, -247.1882629, 247.5271454
2: -162.3865051, 139.2944336, -167.7163086, 142.9577484, -305.3442383, 307.0107422
3: -81.5001144, 157.3161316, -83.0653000, 162.6521912, -244.1523132, 240.3814392
4: -176.2256470, 147.1510620, -181.9947968, 151.0111847, -327.2368164, 329.1458435

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0042211, upper bound: 176.0622054
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0042211, upper bound: 176.0622054
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -133.4149170, 135.3246155, -91.4743271, 112.7514648, -246.1663818, 226.7988892
1: -110.0467529, 123.7133789, -74.9706650, 101.6991348, -211.7458801, 198.6840210
2: -153.9834900, 136.3329620, -105.2589417, 115.1343307, -269.1178284, 241.5918884
3: -79.7415695, 148.9605865, -68.8409653, 104.0475159, -183.7890625, 217.8015442
4: -167.0974121, 144.1869965, -115.4177475, 121.8774719, -288.9748840, 259.6047363

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.8521211, upper bound: 172.8753660
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.2371663, upper bound: 172.2024700
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.1723789, upper bound: 171.2463650
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -146.9546051, 144.1370697, -105.0934448, 119.2186661, -266.1732788, 249.2304840
1: -121.2054749, 131.9143066, -86.9803162, 108.7176056, -229.9230652, 218.8946228
2: -169.7323761, 143.6659851, -122.0204010, 120.9229431, -290.6552734, 265.6863708
3: -83.4981003, 164.1896667, -71.7856598, 121.2669067, -204.7649994, 235.9753265
4: -184.1379089, 151.7879486, -132.6455536, 127.7139664, -311.8518677, 284.4335022

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.9715486, upper bound: 191.0013341
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.5665355, upper bound: 187.9702720
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6871129, upper bound: 190.7946489
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -105.0934448, 119.2186661, -146.9546051, 144.1370697, -249.2304993, 266.1732788
1: -86.9803162, 108.7176056, -121.2054749, 131.9143066, -218.8946228, 229.9230652
2: -122.0204010, 120.9229431, -169.7323761, 143.6659851, -265.6863708, 290.6552734
3: -71.7856598, 121.2669067, -83.4981003, 164.1896667, -235.9753265, 204.7649994
4: -132.6455536, 127.7139664, -184.1379089, 151.7879486, -284.4335022, 311.8518677

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.9986573, upper bound: 190.8600319
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.8979543, upper bound: 190.8600319
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -105.0934448, 119.2186661, -109.4849014, 122.1597290, -227.2531586, 228.7035675
1: -86.9803162, 108.7176056, -90.5823822, 111.4376678, -198.4179840, 199.2999573
2: -122.0204010, 120.9229431, -127.0952301, 124.0488815, -246.0692749, 248.0181732
3: -71.7856598, 121.2669067, -73.2953796, 126.1313477, -197.9170074, 194.5622864
4: -132.6455536, 127.7139664, -138.1383820, 131.0992584, -263.7448120, 265.8522949

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0041011, upper bound: 191.0006421
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0040535, upper bound: 191.0006421
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0040535, upper bound: 191.0006421
time: 0.58 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.17 seconds
IS_A1_B1_B1_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.17
Output dim: 0, lower bound: -175.9325601, upper bound: 175.9325601
IS_A1_B1_B1_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.17
Output dim: 0, lower bound: -175.9325601, upper bound: 175.9325601
IS_A1_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -175.8536255, upper bound: 176.4882298
IS_A1_B1_B1_A2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.17
Output dim: 0, lower bound: -175.7003913, upper bound: 175.6980359
IS_A1_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -173.8339612, upper bound: 176.1229116
IS_A1_B1_B2_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.17
Output dim: 0, lower bound: -171.9810240, upper bound: 169.1308873
IS_A1_B1_B2_A2_A1_A1, status: Status.VERIFIED, split count: 6, time: 5.17
Output dim: 0, lower bound: -172.5392970, upper bound: 175.0792237
IS_A1_B1_B2_A2_A1_A2, status: Status.VERIFIED, split count: 6, time: 5.17
Output dim: 0, lower bound: -169.6473842, upper bound: 173.4643507
IS_A1_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -178.0171832, upper bound: 181.5145253
IS_A1_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -177.5617945, upper bound: 181.4154661
IS_A1_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -176.4909771, upper bound: 176.2230518
IS_A1_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -176.4909771, upper bound: 176.4660804
IS_A1_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -176.4909771, upper bound: 176.2230518
IS_A1_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -176.4909771, upper bound: 176.4660804
IS_A1_B2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.17
Output dim: 0, lower bound: -176.0620677, upper bound: 176.0620666
IS_A1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -176.0620677, upper bound: 176.3228313
IS_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -176.3228325, upper bound: 176.0996675
IS_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -176.3228325, upper bound: 176.3630985
IS_A1_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -178.8084666, upper bound: 183.2321584
IS_A1_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -179.0247094, upper bound: 183.3221738
IS_A1_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -178.6941563, upper bound: 183.2355908
IS_A1_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -178.6939077, upper bound: 183.2355940
IS_A2_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -183.2321584, upper bound: 178.8084666
IS_A2_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -183.3221738, upper bound: 179.0247094
IS_A2_B1_B2_A1_B1_B1, status: Status.VERIFIED, split count: 6, time: 5.17
Output dim: 0, lower bound: -175.0792237, upper bound: 172.5392970
IS_A2_B1_B2_A1_B1_B2, status: Status.VERIFIED, split count: 6, time: 5.17
Output dim: 0, lower bound: -173.4643507, upper bound: 169.6473842
IS_A2_B1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -181.5145253, upper bound: 178.0171832
IS_A2_B1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -181.4154627, upper bound: 177.5617945
IS_A2_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -183.2370949, upper bound: 178.6935603
IS_A2_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -183.2370949, upper bound: 178.6935068
IS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.17
Output dim: 0, lower bound: -176.0042211, upper bound: 176.0622054
IS_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.17
Output dim: 0, lower bound: -176.0042211, upper bound: 176.0622054
IS_A2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -178.2371663, upper bound: 172.2024700
IS_A2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -176.1723789, upper bound: 171.2463650
IS_A2_B2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -190.5665355, upper bound: 187.9702720
IS_A2_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -190.6871129, upper bound: 190.7946489
IS_A2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -190.9986573, upper bound: 190.8600319
IS_A2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -190.8979543, upper bound: 190.8600319
IS_A2_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -191.0040535, upper bound: 191.0006421
IS_A2_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.17
Output dim: 0, lower bound: -191.0040535, upper bound: 191.0006421

## BFS IS instance: IS_A1_B1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -145.4437866, 155.1196899, -175.5351257, 172.2845612, -317.7283325, 330.6548157
1: -120.2910233, 142.7280731, -144.8220978, 158.6592407, -278.9502563, 287.5501709
2: -168.8097839, 153.7775879, -202.7999878, 169.9971008, -338.8068848, 356.5775757
3: -84.3281860, 166.2122345, -93.4979782, 196.0755005, -280.4036865, 259.7102051
4: -183.0110016, 162.3237000, -219.9038544, 179.5206146, -362.5314941, 382.2275391

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -172.2457604, upper bound: 175.1151193
time: 0.67 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1586552, upper bound: 175.6476090
time: 0.63 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.7003913, upper bound: 175.6980359
time: 0.67 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.7003913, upper bound: 175.6980359
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -169.1473541, 166.2080994, -137.9302368, 138.2968903, -307.4442139, 304.1383057
1: -139.5300598, 152.7494812, -113.8378830, 126.4625702, -265.9925842, 266.5873718
2: -195.3223114, 164.0295563, -159.3600159, 138.0276642, -333.3498840, 323.3895569
3: -91.0397263, 187.8306580, -80.7683945, 154.8077545, -245.8474731, 268.5990601
4: -211.7273102, 173.3522797, -172.9420471, 145.7950592, -357.5223083, 346.2943115

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_A1_A1_B1_A1

### Relational analysis result of IS_A1_B1_B2_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -166.7471673, upper bound: 168.5655393
time: 0.64 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A1_A1_B1_A1

### Relational analysis result of IS_A1_B1_B2_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.9810240, upper bound: 169.1304536
time: 0.65 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_B1_A2

### Relational analysis result of IS_A1_B1_B2_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.9810240, upper bound: 169.1308832
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -129.6816864, 144.5743561, -139.1485138, 139.5306702, -269.2123413, 283.7228699
1: -107.5782166, 132.7650146, -114.8711166, 127.6223602, -235.2005768, 247.6361237
2: -150.8875885, 143.4069061, -160.7460327, 139.6595001, -290.5470886, 304.1529541
3: -79.7009277, 149.3589630, -81.4494858, 156.3122864, -236.0132141, 230.8084412
4: -163.1866455, 151.6977539, -174.5375061, 147.5945435, -310.7811279, 326.2352600

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B2_A2_A2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.9815962, upper bound: 181.1793392
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A2_A2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9301834, upper bound: 180.9512748
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -169.3302704, upper bound: 170.9149021
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -135.7113647, 148.2635193, -141.5959320, 140.7404175, -276.4517822, 289.8594360
1: -112.2441559, 136.1781921, -116.8814926, 128.7644958, -241.0086517, 253.0596924
2: -157.4493561, 147.1098328, -163.5976257, 140.6518860, -298.1012268, 310.7073975
3: -81.6211243, 154.4183350, -81.9615936, 158.4699860, -240.0911102, 236.3799286
4: -170.7235718, 155.3350983, -177.4522095, 148.6237946, -319.3473511, 332.7872620

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_A2_A2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.5617945, upper bound: 181.4154627
time: 0.68 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.5617945, upper bound: 181.4154627
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -134.5135040, 149.5867004, -134.1971741, 148.7945404, -283.3080139, 283.7838745
1: -111.5995026, 137.4353180, -111.3474274, 136.7601166, -248.3596191, 248.7827454
2: -156.6116943, 148.4280853, -156.3009338, 147.6230621, -304.2347412, 304.7290039
3: -81.5965118, 154.7332458, -81.5041962, 155.0912476, -236.6877594, 236.2374420
4: -169.4080505, 156.8603821, -169.0137177, 155.9085236, -325.3165588, 325.8740845

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -172.3928601, upper bound: 174.5433831
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.1980828, upper bound: 175.9356963
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -134.5135040, 149.5867004, -163.4789886, 169.5801392, -304.0936279, 313.0656738
1: -111.5995026, 137.4353180, -135.0603180, 156.7100983, -268.3096008, 272.4956360
2: -156.6116943, 148.4280853, -189.7915955, 169.0911255, -325.7027893, 338.2196045
3: -81.5965118, 154.7332458, -91.0852814, 187.0558929, -268.6523438, 245.8184967
4: -169.4080505, 156.8603821, -205.9918976, 178.0162506, -347.4243164, 362.8522644

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -173.1242715, upper bound: 172.6205713
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.1980829, upper bound: 176.1468582
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -168.1466064, 173.6803131, -134.1971741, 148.7945404, -316.9411621, 307.8775024
1: -138.6561737, 160.4846039, -111.3474274, 136.7601166, -275.4162903, 271.8320312
2: -194.8035889, 173.0832062, -156.3009338, 147.6230621, -342.4266357, 329.3841553
3: -92.7270432, 191.0919342, -81.5041962, 155.0912476, -247.8182831, 272.5961304
4: -211.6338806, 182.2114868, -169.0137177, 155.9085236, -367.5424194, 351.2251892

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -172.4305643, upper bound: 174.5389589
time: 0.56 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.1618969, upper bound: 175.9162847
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -168.1466064, 173.6803131, -163.4789886, 169.5801392, -337.7267456, 337.1593018
1: -138.6561737, 160.4846039, -135.0603180, 156.7100983, -295.3662720, 295.5449219
2: -194.8035889, 173.0832062, -189.7915955, 169.0911255, -363.8947144, 362.8747559
3: -92.7270432, 191.0919342, -91.0852814, 187.0558929, -279.7828674, 282.1772156
4: -211.6338806, 182.2114868, -205.9918976, 178.0162506, -389.6501160, 388.2033386

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -173.0977734, upper bound: 171.1347229
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.1618970, upper bound: 175.9948914
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -138.8819580, 152.1155396, -163.9227905, 169.8754120, -308.7573547, 316.0383301
1: -115.0018082, 139.7933044, -135.4270020, 157.0023041, -272.0041199, 275.2203064
2: -161.3812408, 150.9663696, -190.3155060, 169.3701782, -330.7514038, 341.2818604
3: -83.0931854, 158.2641449, -91.2118607, 187.4907074, -270.5838928, 249.4759979
4: -174.8187408, 159.3495178, -206.5502625, 178.3105011, -353.1292419, 365.8997498

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -172.6988546, upper bound: 172.7105970
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.7602245, upper bound: 175.9898366
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -168.0211487, 173.2003021, -136.3991699, 150.2267609, -318.2479248, 309.5994568
1: -138.5490112, 159.9856415, -113.0783310, 138.1056061, -276.6546021, 273.0639648
2: -194.7471924, 172.6358185, -158.7032776, 149.0285645, -343.7757568, 331.3391113
3: -92.8779449, 190.8125153, -82.0881042, 157.0976105, -249.9755554, 272.9005737
4: -211.6457520, 181.7321167, -171.8273163, 157.3073120, -368.9530029, 353.5594482

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -172.7105980, upper bound: 174.4338924
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9898378, upper bound: 175.7848856
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -168.0211487, 173.2003021, -163.9227905, 169.8754120, -337.8965454, 337.1231079
1: -138.5490112, 159.9856415, -135.4270020, 157.0023041, -295.5513000, 295.4125977
2: -194.7471924, 172.6358185, -190.3155060, 169.3701782, -364.1173706, 362.9513245
3: -92.8779449, 190.8125153, -91.2118607, 187.4907074, -280.3686523, 282.0242920
4: -211.6457520, 181.7321167, -206.5502625, 178.3105011, -389.9562378, 388.2823792

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -172.7105980, upper bound: 174.6802283
time: 0.82 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9898378, upper bound: 175.9948915
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -177.7269897, 173.5277405, -81.8415833, 104.8546906, -282.5816650, 255.3693237
1: -146.6130524, 159.8040771, -68.0090408, 94.7844849, -241.3975372, 227.8131104
2: -205.3420410, 171.1296082, -95.3412018, 107.1040955, -312.4461365, 266.4707947
3: -94.0076370, 197.8811493, -64.9595795, 96.7800903, -190.7877197, 262.8407288
4: -222.5492554, 180.8058014, -103.6640930, 113.4794769, -336.0287170, 284.4698792

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.3997737, upper bound: 182.7084059
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.8084671, upper bound: 183.2321616
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.1004555, upper bound: 181.3773812
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.8084666, upper bound: 183.2321584
time: 0.75 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.3103504, upper bound: 182.9562698
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -182.9076538, 176.9971466, -98.1761932, 114.4870758, -297.3947144, 275.1733398
1: -150.8231964, 163.0276184, -81.2362823, 104.1438599, -254.9669952, 244.2638855
2: -211.2644501, 174.4967804, -113.9444351, 116.7703171, -328.0347595, 288.4411621
3: -95.7824173, 203.4107056, -69.7811661, 113.5294724, -209.3118439, 273.1918640
4: -229.0425720, 184.3425751, -123.9752121, 123.4040375, -352.4465332, 308.3177795

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.9769144, upper bound: 183.3103472
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -171.4643360, upper bound: 178.4317657
time: 0.77 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -171.4643354, upper bound: 181.2325348
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -150.0166016, 158.1433105, -95.6446609, 113.3904877, -263.4070435, 253.7879486
1: -124.0767059, 145.5217438, -79.2515564, 103.1769333, -227.2536316, 224.7732849
2: -174.1481781, 156.6505737, -111.1647339, 115.6497650, -289.7979126, 267.8152771
3: -86.0277176, 171.3147736, -69.2544556, 111.2451477, -197.2728424, 240.5692291
4: -188.7997437, 165.3383484, -120.8667679, 122.2402802, -311.0399780, 286.2051086

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B2_A2_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.6939072, upper bound: 183.2355908
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.6939072, upper bound: 183.2355908
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -142.8482971, 153.7097778, -122.8770905, 132.3203735, -275.1686707, 276.5868225
1: -118.2187424, 141.4161072, -101.2287979, 121.3913574, -239.6100922, 242.6448975
2: -165.9113617, 152.4881897, -142.3907928, 133.3691559, -299.2804260, 294.8789673
3: -83.7646942, 163.5688019, -77.8224792, 140.9369202, -224.7016144, 241.3912506
4: -179.8201904, 161.0068207, -155.2873383, 140.4335175, -320.2537231, 316.2941589

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -171.9853837, upper bound: 179.1460191
time: 0.74 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -171.9853837, upper bound: 180.6288913
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -81.8415833, 104.8546906, -177.7315369, 173.5907593, -255.4323425, 282.5862427
1: -68.0090408, 94.7844849, -146.6177979, 159.8526306, -227.8616638, 241.4022827
2: -95.3412018, 107.1040955, -205.3540955, 171.1800842, -266.5212402, 312.4581604
3: -64.9595795, 96.7800903, -94.0161057, 197.9216156, -262.8811646, 190.7961884
4: -103.6640930, 113.4794769, -222.5637665, 180.8705292, -284.5346069, 336.0432434

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.7084059, upper bound: 178.3997737
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.2321366, upper bound: 178.8084420
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.3773807, upper bound: 178.1004555
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.2321584, upper bound: 178.8084666
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.9562694, upper bound: 178.3103503
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -98.1761932, 114.4870758, -182.9138031, 177.0833893, -275.2595520, 297.4008789
1: -81.2362823, 104.1438599, -150.8296814, 163.0941467, -244.3304138, 254.9734650
2: -113.9444351, 116.7703171, -211.2808533, 174.5656891, -288.5101013, 328.0511169
3: -69.7811661, 113.5294724, -95.7939606, 203.4659576, -273.2471313, 209.3234100
4: -123.9752121, 123.4040375, -229.0622559, 184.4309082, -308.4061279, 352.4663086

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.3103472, upper bound: 178.9769144
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.4317654, upper bound: 173.6777556
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.4317677, upper bound: 179.0247098
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -141.2733459, 140.5992279, -129.6816864, 144.5743561, -285.8477173, 270.2809143
1: -116.5637589, 128.5899658, -107.5782166, 132.7650146, -249.3287659, 236.1681824
2: -163.1380310, 140.6365051, -150.8875885, 143.4069061, -306.5448914, 291.5240784
3: -82.0275269, 158.2847137, -79.7009277, 149.3589630, -231.3864899, 237.9856415
4: -177.1408997, 148.6402893, -163.1866455, 151.6977539, -328.8385925, 311.8269043

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.1793399, upper bound: 177.9815962
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.9512748, upper bound: 176.9301834
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -170.9149021, upper bound: 169.3302704
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -144.2499084, 142.0549011, -135.7113647, 148.2635193, -292.5134277, 277.7662659
1: -118.9967957, 129.9512482, -112.2441559, 136.1781921, -255.1749878, 242.1954041
2: -166.5941162, 141.8513794, -157.4493561, 147.1098328, -313.7039185, 299.3007202
3: -82.6747971, 160.9079895, -81.6211243, 154.4183350, -237.0930786, 242.5291138
4: -180.7047577, 149.9111481, -170.7235718, 155.3350983, -336.0398560, 320.6347046

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.4154633, upper bound: 177.5617945
time: 0.91 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.4154633, upper bound: 177.5617945
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -95.6446609, 113.3904877, -150.0166016, 158.1433105, -253.7879333, 263.4071045
1: -79.2515564, 103.1769333, -124.0767059, 145.5217438, -224.7733002, 227.2536163
2: -111.1647339, 115.6497650, -174.1481781, 156.6505737, -267.8152771, 289.7979126
3: -69.2544556, 111.2451477, -86.0277176, 171.3147736, -240.5691986, 197.2728577
4: -120.8667679, 122.2402802, -188.7997437, 165.3383484, -286.2051086, 311.0399780

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_A2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.2370949, upper bound: 178.6935068
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.2370949, upper bound: 178.6935068
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -122.8770905, 132.3203735, -142.8482971, 153.7097778, -276.5868225, 275.1686707
1: -101.2287979, 121.3913574, -118.2187424, 141.4161072, -242.6448975, 239.6100769
2: -142.3907928, 133.3691559, -165.9113617, 152.4881897, -294.8789673, 299.2804565
3: -77.8224792, 140.9369202, -83.7646942, 163.5688019, -241.3912811, 224.7016144
4: -155.2873383, 140.4335175, -179.8201904, 161.0068207, -316.2941589, 320.2537231

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B2_A2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.0305871, upper bound: 174.2542615
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.0305907, upper bound: 178.6935073
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -120.2844696, 127.7973938, -86.9976425, 110.3255157, -230.6099854, 214.7949982
1: -99.2190857, 116.1937943, -71.3080673, 99.2864151, -198.5054932, 187.5018616
2: -138.8355865, 129.2335205, -100.0520935, 112.6481018, -251.4836884, 229.2856140
3: -76.1732635, 135.3749237, -67.7097015, 99.2617035, -175.4349670, 203.0846252
4: -150.9302673, 136.7985229, -109.7414093, 119.3991241, -270.3293457, 246.5399017

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.2371663, upper bound: 172.2024700
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.7862264, upper bound: 171.2463636
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.7862264, upper bound: 171.2463650
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -121.2768097, 126.9378815, -89.0439682, 111.2607880, -232.5375824, 215.9818420
1: -100.1170502, 115.2286682, -72.9598236, 100.2111053, -200.3281555, 188.1884918
2: -139.8791656, 128.4387054, -102.4119720, 113.6473694, -253.5265198, 230.8506775
3: -76.1534882, 134.5812073, -68.1857986, 101.1135864, -177.2670746, 202.7669983
4: -151.7111359, 136.0404205, -112.3522949, 120.3452454, -272.0563354, 248.3927002

Time for backsubstitution: 2.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.1723796, upper bound: 171.2463641
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.7862260, upper bound: 171.2463636
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.7862264, upper bound: 171.2463650
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -141.2733459, 140.5992279, -90.5509186, 110.2088623, -251.4821930, 231.1501465
1: -116.5637589, 128.5899658, -75.2615891, 99.7704315, -216.3341980, 203.8515625
2: -163.1380310, 140.6365051, -105.4250793, 112.4691238, -275.6070862, 246.0615845
3: -82.0275269, 158.2847137, -67.7298965, 105.9263000, -187.9537964, 226.0146179
4: -177.1408997, 148.6402893, -114.2826920, 119.2104034, -296.3512878, 262.9229736

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_B2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.5135831, upper bound: 187.9700798
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.5135831, upper bound: 187.9702720
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -144.2499084, 142.0549011, -94.7659683, 112.1141891, -256.3640442, 236.8208313
1: -118.9967957, 129.9512482, -78.4471359, 101.6716919, -220.6684875, 208.3983459
2: -166.5941162, 141.8513794, -109.9604187, 114.4650192, -281.0591125, 251.8117981
3: -82.6747971, 160.9079895, -68.8886490, 108.8614044, -191.5361633, 229.7966309
4: -180.7047577, 149.9111481, -119.6341171, 121.0348969, -301.7396240, 269.5452576

Time for backsubstitution: 2.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6341706, upper bound: 190.7613091
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6341706, upper bound: 190.7946500
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -81.8415833, 104.8546906, -140.2901459, 139.6544342, -221.4960175, 245.1448212
1: -68.0090408, 94.7844849, -115.7728958, 127.7539825, -195.7630157, 210.5573730
2: -95.3412018, 107.1040955, -162.0565033, 139.4447937, -234.7859497, 269.1605835
3: -64.9595795, 96.7800903, -81.3221893, 156.8657684, -221.8253326, 178.1022339
4: -103.6640930, 113.4794769, -175.7476807, 147.3850098, -251.0491028, 289.2271423

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7469472, upper bound: 190.5250163
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7919052, upper bound: 190.5833194
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -98.1761932, 114.4870758, -146.9546051, 144.1370697, -242.3132477, 261.4416809
1: -81.2362823, 104.1438599, -121.2054749, 131.9143066, -213.1505737, 225.3493195
2: -113.9444351, 116.7703171, -169.7323761, 143.6659851, -257.6103516, 286.5026550
3: -69.7811661, 113.5294724, -83.4981003, 164.1896667, -233.9708252, 197.0275421
4: -123.9752121, 123.4040375, -184.1379089, 151.7879486, -275.7631531, 307.5419312

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.2427027, upper bound: 189.7198607
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3648142, upper bound: 187.9250716
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7031419, upper bound: 190.5833198
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -95.6446609, 113.3904877, -108.4479370, 121.5212555, -217.1658630, 221.8384247
1: -79.2515564, 103.1769333, -89.7393570, 110.8410110, -190.0925598, 192.9162903
2: -111.1647339, 115.6497650, -125.9127426, 123.4667587, -234.6314850, 241.5625000
3: -69.2544556, 111.2451477, -72.9947433, 125.0228806, -194.2773438, 184.2398834
4: -120.8667679, 122.2402802, -136.8418732, 130.4915009, -251.3582764, 259.0820923

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0040538, upper bound: 191.0006421
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0040538, upper bound: 191.0006421
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -122.8770905, 132.3203735, -100.3812408, 116.7194290, -239.5964966, 232.7015991
1: -101.2287979, 121.3913574, -83.0567322, 106.1177826, -207.3465881, 204.4480591
2: -142.3907928, 133.3691559, -116.5011902, 118.9823761, -261.3731689, 249.8702850
3: -77.8224792, 140.9369202, -70.7747192, 116.1197510, -193.9422150, 211.7116394
4: -155.2873383, 140.4335175, -126.7332535, 125.8554764, -281.1427917, 267.1667175

Time for backsubstitution: 2.15 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=279.9653625488281
rel_dist={0: [-191.43347747308115, 191.4334774730812]}

## Binary search (step 2) starts
Candidate diff: 0.0625000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.7668396, upper bound: 180.8103516
time: 0.79 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1338263, upper bound: 191.1338286
time: 0.61 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.59 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 0, lower bound: -178.7668396, upper bound: 180.8103516
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 0, lower bound: -191.1338263, upper bound: 191.1338286

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -165.7317810, 169.9891968, -99.5717773, 119.4847870, -285.2165527, 269.5609436
1: -136.9342194, 156.5604401, -82.2094345, 108.2213745, -245.1555939, 238.7698669
2: -192.2410583, 168.2551727, -115.4599609, 121.4488373, -313.6898804, 283.7151489
3: -91.3198547, 188.0007629, -71.8553314, 115.1565094, -206.4763641, 259.8560486
4: -208.4816437, 177.5048828, -125.8874283, 128.5055542, -336.9871826, 303.3922424

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.2946476, upper bound: 180.6356403
time: 0.59 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.2686537, upper bound: 180.6270358
time: 0.65 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -126.3109131, 134.7957306, -135.6846161, 140.4945374, -266.8054504, 270.4803467
1: -104.3817749, 123.2117767, -112.0749969, 128.5360260, -232.9178009, 235.2867432
2: -146.5827942, 135.2793121, -157.4221497, 140.3601532, -286.9429321, 292.7014771
3: -78.7137909, 144.2006378, -81.2042694, 154.1939697, -232.9077301, 225.4048920
4: -159.2684784, 142.8177032, -170.9760895, 148.0922089, -307.3606873, 313.7937927

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.8103516, upper bound: 178.7668396
time: 0.59 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.8103516, upper bound: 191.1338286
time: 0.64 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.48 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.48
Output dim: 0, lower bound: -178.2946476, upper bound: 180.6356403
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.48
Output dim: 0, lower bound: -178.2686537, upper bound: 180.6270358
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.48
Output dim: 0, lower bound: -180.8103516, upper bound: 178.7668396
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.48
Output dim: 0, lower bound: -180.8103516, upper bound: 191.1338286

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -157.4777222, 164.4136200, -116.7327271, 126.4945068, -283.9722290, 281.1463623
1: -130.2530823, 151.3745728, -96.3356018, 114.8215714, -245.0746460, 247.7101746
2: -182.8538666, 162.8088531, -134.7284698, 127.7783737, -310.6322327, 297.5372925
3: -88.6713257, 179.2406311, -75.4720917, 131.6392822, -220.3106079, 254.7127228
4: -198.0883026, 171.8528290, -146.4232788, 135.2839661, -333.3722534, 318.2761230

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2603986, upper bound: 177.4022479
time: 0.66 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2603986, upper bound: 180.6356403
time: 0.54 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -157.3497009, 163.5761108, -85.2214203, 109.8894043, -267.2391052, 248.7975311
1: -130.0755157, 150.5910950, -70.3995972, 98.9607468, -229.0362549, 220.9906921
2: -182.5816956, 161.9783936, -98.7515717, 111.9022064, -294.4838867, 260.7299194
3: -88.4746704, 179.0487671, -67.3120422, 99.8246231, -188.2992706, 246.3607788
4: -197.9725342, 170.9270325, -107.9173965, 118.5906830, -316.5632019, 278.8444214

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2237073, upper bound: 177.2237069
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2237073, upper bound: 180.6270358
time: 0.61 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -126.3109131, 134.7957306, -165.7308350, 169.9760437, -296.2869568, 300.5265503
1: -104.3817749, 123.2117767, -136.9331818, 156.5503845, -260.9321594, 260.1449585
2: -146.5827942, 135.2793121, -192.2384796, 168.2447968, -314.8275757, 327.5177917
3: -78.7137909, 144.2006378, -91.3181000, 187.9923859, -266.7061462, 235.5187378
4: -159.2684784, 142.8177032, -208.4785309, 177.4915466, -336.7600098, 351.2962341

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.6586640, upper bound: 178.4457772
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.6270325, upper bound: 178.2686523
time: 0.72 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -126.3109131, 134.7957306, -126.3109131, 134.7957306, -261.1066284, 261.1066284
1: -104.3817749, 123.2117767, -104.3817749, 123.2117767, -227.5935364, 227.5935211
2: -146.5827942, 135.2793121, -146.5827942, 135.2793121, -281.8620911, 281.8620911
3: -78.7137909, 144.2006378, -78.7137909, 144.2006378, -222.9144287, 222.9144287
4: -159.2684784, 142.8177032, -159.2684784, 142.8177032, -302.0861816, 302.0861816

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.6356403, upper bound: 191.0581866
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.6270358, upper bound: 191.0589675
time: 0.69 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.80 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.80
Output dim: 0, lower bound: -177.2603986, upper bound: 177.4022479
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.80
Output dim: 0, lower bound: -177.2603986, upper bound: 180.6356403
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.80
Output dim: 0, lower bound: -177.2237073, upper bound: 177.2237069
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.80
Output dim: 0, lower bound: -177.2237073, upper bound: 180.6270358
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.80
Output dim: 0, lower bound: -180.6586640, upper bound: 178.4457772
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.80
Output dim: 0, lower bound: -180.6270325, upper bound: 178.2686523
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.80
Output dim: 0, lower bound: -180.6356403, upper bound: 191.0581866
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.80
Output dim: 0, lower bound: -180.6270358, upper bound: 191.0589675

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -157.4777222, 164.4136200, -178.0234985, 173.9511871, -331.4288940, 342.4371033
1: -130.2530823, 151.3745728, -147.1725464, 160.3473358, -290.6004028, 298.5471191
2: -182.8538666, 162.8088531, -205.9979858, 171.7688446, -354.6226196, 368.8067932
3: -88.6713257, 179.2406311, -94.6885681, 199.3521423, -288.0234680, 273.9291382
4: -198.0883026, 171.8528290, -223.0717163, 181.1529694, -379.2412720, 394.9245300

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2603986, upper bound: 177.4022479
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2603986, upper bound: 177.4022479
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -157.4777222, 164.4136200, -137.9891357, 138.4441223, -295.9218140, 302.4026794
1: -130.2530823, 151.3745728, -113.9848480, 126.7239685, -256.9770203, 265.3593750
2: -182.8538666, 162.8088531, -159.5431213, 139.2382050, -322.0920715, 322.3519897
3: -88.6713257, 179.2406311, -81.2419891, 155.2761383, -243.9474640, 260.4826050
4: -198.0883026, 171.8528290, -173.0452881, 147.0822449, -345.1705322, 344.8980713

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2603986, upper bound: 180.6356403
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2603986, upper bound: 180.6356403
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -157.3497009, 163.5761108, -141.8359833, 152.0677795, -309.4174500, 305.4120789
1: -130.0755157, 150.5910950, -117.7377014, 140.2324524, -270.3079529, 268.3287659
2: -182.5816956, 161.9783936, -165.1059723, 151.2542572, -333.8359375, 327.0843506
3: -88.4746704, 179.0487671, -82.6524429, 162.8121643, -251.2868042, 261.7012024
4: -197.9725342, 170.9270325, -178.3360138, 159.5032654, -357.4757996, 349.2630615

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.4478720, upper bound: 176.4004234
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3287051, upper bound: 176.3287039
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -157.3497009, 163.5761108, -104.7171173, 119.2199860, -276.5697021, 268.2932129
1: -130.0755157, 150.5910950, -86.7094498, 108.3716736, -238.4471893, 237.3005371
2: -182.5816956, 161.9783936, -121.6465912, 121.3357544, -303.9174500, 283.6249390
3: -88.4746704, 179.0487671, -71.8716888, 121.1719818, -209.6466522, 250.9204559
4: -197.9725342, 170.9270325, -132.2137604, 128.2694855, -326.2420044, 303.1408081

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2213328, upper bound: 180.6270358
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2211052, upper bound: 180.6264735
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -117.3274231, 128.7620850, -182.9076538, 176.9971466, -294.3245850, 311.6697388
1: -97.1327438, 117.6113586, -150.8231964, 163.0276184, -260.1603699, 268.4345703
2: -136.3496246, 129.8052979, -211.2644501, 174.4967804, -310.8464050, 341.0697327
3: -75.9192734, 134.5744934, -95.7824173, 203.4107056, -279.3299561, 230.3568878
4: -148.0383911, 137.1970367, -229.0425720, 184.3425751, -332.3809204, 366.2395630

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.6133758, upper bound: 178.2547482
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.6133758, upper bound: 178.2547482
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -116.7671204, 127.6062851, -150.8867340, 158.6611786, -275.4282837, 278.4929810
1: -96.5490570, 116.5216522, -124.7843018, 146.0037994, -242.5528564, 241.3059387
2: -135.5279083, 128.8826294, -175.1371307, 157.1483612, -292.6762695, 304.0197754
3: -75.6287994, 133.9197845, -86.2978363, 172.2207794, -247.8495789, 220.2176056
4: -147.2747650, 136.1374817, -189.8798828, 165.8626709, -313.1374207, 326.0173645

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.6264701, upper bound: 178.2684349
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.6264701, upper bound: 178.2648402
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -146.9546051, 144.1370697, -117.3274231, 128.7620850, -275.7166748, 261.4644775
1: -121.2054749, 131.9143066, -97.1327438, 117.6113586, -238.8168335, 229.0470581
2: -169.7323761, 143.6659851, -136.3496246, 129.8052979, -299.5376282, 280.0156250
3: -83.4981003, 164.1896667, -75.9192734, 134.5744934, -218.0726013, 240.1089478
4: -184.1379089, 151.7879486, -148.0383911, 137.1970367, -321.3349304, 299.8263245

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0331285, upper bound: 191.0581782
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0332264, upper bound: 191.0332264
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0332264, upper bound: 191.0581841
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -109.4849014, 122.1597290, -116.7671204, 127.6062851, -237.0911865, 238.9268494
1: -90.5823822, 111.4376678, -96.5490570, 116.5216522, -207.1040039, 207.9867096
2: -127.0952301, 124.0488815, -135.5279083, 128.8826294, -255.9778595, 259.5767822
3: -73.2953796, 126.1313477, -75.6287994, 133.9197845, -207.2151489, 201.7601471
4: -138.1383820, 131.0992584, -147.2747650, 136.1374817, -274.2758179, 278.3740234

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0585075, upper bound: 191.0532105
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0588960, upper bound: 191.0589650
time: 0.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.68 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -177.2603986, upper bound: 177.4022479
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -177.2603986, upper bound: 177.4022479
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -177.2603986, upper bound: 180.6356403
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -177.2603986, upper bound: 180.6356403
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -176.4478720, upper bound: 176.4004234
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -176.3287051, upper bound: 176.3287039
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -177.2213328, upper bound: 180.6270358
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -177.2211052, upper bound: 180.6264735
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -180.6133758, upper bound: 178.2547482
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -180.6133758, upper bound: 178.2547482
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -180.6264701, upper bound: 178.2684349
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -180.6264701, upper bound: 178.2648402
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -191.0332264, upper bound: 191.0332264
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -191.0332264, upper bound: 191.0581841
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -191.0585075, upper bound: 191.0532105
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -191.0588960, upper bound: 191.0589650

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -182.9076538, 176.9971466, -178.0234985, 173.9511871, -356.8588257, 355.0206299
1: -150.8231964, 163.0276184, -147.1725464, 160.3473358, -311.1705322, 310.2001648
2: -211.2644501, 174.4967804, -205.9979858, 171.7688446, -383.0332642, 380.4947205
3: -95.7824173, 203.4107056, -94.6885681, 199.3521423, -295.1345520, 298.0992737
4: -229.0425720, 184.3425751, -223.0717163, 181.1529694, -410.1954346, 407.4142761

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -168.2944111, upper bound: 170.0480897
time: 0.85 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -167.6340643, upper bound: 168.1179708
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -149.8873901, 157.9709930, -178.0234985, 173.9511871, -323.8385620, 335.9945068
1: -123.9680481, 145.3655548, -147.1725464, 160.3473358, -284.3153687, 292.5380859
2: -173.9833679, 156.4939575, -205.9979858, 171.7688446, -345.7521362, 362.4918823
3: -85.9076080, 171.1465759, -94.6885681, 199.3521423, -285.2597656, 265.8351440
4: -188.5898285, 165.1896057, -223.0717163, 181.1529694, -369.7427673, 388.2613220

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -172.6433144, upper bound: 174.9487920
time: 0.65 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9238377, upper bound: 177.0821615
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -182.9076538, 176.9971466, -137.9891357, 138.4441223, -321.3517761, 314.9862671
1: -150.8231964, 163.0276184, -113.9848480, 126.7239685, -277.5471497, 277.0124512
2: -211.2644501, 174.4967804, -159.5431213, 139.2382050, -350.5026550, 334.0399170
3: -95.7824173, 203.4107056, -81.2419891, 155.2761383, -251.0585022, 284.6526794
4: -229.0425720, 184.3425751, -173.0452881, 147.0822449, -376.1248169, 357.3878174

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.2392459, upper bound: 172.3459127
time: 0.87 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -168.7579639, upper bound: 170.5839328
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -149.8873901, 157.9709930, -137.9891357, 138.4441223, -288.3315125, 295.9601440
1: -123.9680481, 145.3655548, -113.9848480, 126.7239685, -250.6920166, 259.3503723
2: -173.9833679, 156.4939575, -159.5431213, 139.2382050, -313.2214966, 316.0370789
3: -85.9076080, 171.1465759, -81.2419891, 155.2761383, -241.1837463, 252.3885651
4: -188.5898285, 165.1896057, -173.0452881, 147.0822449, -335.6720581, 338.2348633

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.2392459, upper bound: 172.3459127
time: 0.64 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -168.7579630, upper bound: 170.5839296
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -139.9608459, 152.1244659, -134.0693359, 147.2388306, -287.1996155, 286.1937866
1: -116.0446396, 139.7944031, -111.4762650, 135.5263672, -251.5709991, 251.2706299
2: -162.7900696, 150.7699738, -156.3461761, 146.4435577, -309.2336426, 307.1160889
3: -83.2367554, 160.5757751, -80.6929779, 154.5202789, -237.7570343, 241.2687531
4: -176.1137848, 159.3681335, -168.6031647, 154.6781616, -330.7919312, 327.9711914

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3746705, upper bound: 176.3839162
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.4478720, upper bound: 176.4004234
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -144.8820953, 154.9047394, -136.6244202, 148.4888458, -293.3709412, 291.5291443
1: -119.8211594, 142.3726196, -113.3888245, 136.7433777, -256.5645447, 255.7614441
2: -168.0990143, 153.5686188, -158.9859161, 147.7278290, -315.8268433, 312.5545044
3: -84.8019485, 164.4944000, -81.2196503, 156.4427795, -241.2447205, 245.7140503
4: -182.2343140, 162.0848999, -171.7550507, 155.8262329, -338.0605469, 333.8399658

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.0754725, upper bound: 176.2556231
time: 0.71 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3287051, upper bound: 176.3287039
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -146.6312561, 157.2516022, -100.2745743, 116.7616196, -263.3928528, 257.5261536
1: -121.3761597, 144.7100830, -83.0569839, 105.9385452, -227.3146973, 227.7670593
2: -170.4106293, 155.9059601, -116.5230408, 118.8543243, -289.2649536, 272.4289856
3: -85.1553879, 167.9220581, -70.7065887, 116.3929443, -201.5483398, 238.6286316
4: -184.6484985, 164.5325928, -126.6915894, 125.6981735, -310.3466797, 291.2241821

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -172.7533011, upper bound: 171.1108861
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.9572734, upper bound: 180.3311065
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -177.2799835, 178.8639526, -92.0648193, 111.9499054, -289.2298889, 270.9287720
1: -146.0839844, 165.4173737, -76.1331787, 101.4048157, -247.4887848, 241.5505219
2: -205.4298096, 178.1224518, -106.7914429, 114.3367310, -319.7665405, 284.9138794
3: -95.0936050, 201.2472076, -68.6303177, 107.1765137, -202.2701111, 269.8775330
4: -223.2652893, 187.4578400, -116.4129639, 121.1020355, -344.3673096, 303.8707886

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.6099922, upper bound: 179.9655696
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.4252643, upper bound: 179.8622274
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -145.2131958, 143.3489685, -182.9076538, 176.9971466, -322.2103271, 326.2566223
1: -119.8059006, 131.2140198, -150.8231964, 163.0276184, -282.8334656, 282.0372314
2: -167.7163086, 142.9577484, -211.2644501, 174.4967804, -342.2130737, 354.2221985
3: -83.0653000, 162.6521912, -95.7824173, 203.4107056, -286.4760132, 258.4345398
4: -181.9947968, 151.0111847, -229.0425720, 184.3425751, -366.3373108, 380.0537415

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.5075154, upper bound: 170.4498953
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -170.9612000, upper bound: 169.6953763
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -107.5347595, 120.8612366, -182.9076538, 176.9971466, -284.5319214, 303.7688904
1: -88.9876251, 110.2532425, -150.8231964, 163.0276184, -252.0152435, 261.0763855
2: -124.8389893, 122.8079224, -211.2644501, 174.4967804, -299.3357544, 334.0723877
3: -72.6093903, 124.0585861, -95.7824173, 203.4107056, -276.0200806, 219.8409882
4: -135.6271362, 129.8474884, -229.0425720, 184.3425751, -319.9696960, 358.8899841

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -170.4883816, upper bound: 172.5825559
time: 0.56 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.3681035, upper bound: 178.1595677
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -105.1731033, 120.4450989, -145.4621582, 155.3772888, -260.5503845, 265.9072571
1: -87.1250916, 109.8328781, -120.3638306, 142.9510803, -230.0761414, 230.1967163
2: -122.2827759, 122.3578644, -168.9502869, 153.9903564, -276.2731323, 291.3081665
3: -72.3044586, 121.5817719, -84.5951843, 166.4942322, -238.7986908, 206.1769562
4: -132.7809448, 129.3372955, -183.1327515, 162.5378876, -295.3188477, 312.4700317

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -170.6406446, upper bound: 172.4691610
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.3297872, upper bound: 177.9608545
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -136.6329956, 142.3315582, -136.3632660, 149.6460724, -286.2790527, 278.6948242
1: -112.5237198, 130.7263031, -112.8795471, 137.6061859, -250.1299133, 243.6058197
2: -158.3572235, 142.7752991, -158.4175110, 148.5879974, -306.9452209, 301.1928101
3: -82.2227020, 155.9005585, -81.8809204, 156.4909973, -238.7136993, 237.7814789
4: -172.6639557, 150.3804169, -171.7095337, 156.9573822, -329.6213074, 322.0899048

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.5040178, upper bound: 173.2895660
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.3297872, upper bound: 177.9570340
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -146.9546051, 144.1370697, -145.2131958, 143.3489685, -290.3035889, 289.3502197
1: -121.2054749, 131.9143066, -119.8059006, 131.2140198, -252.4194946, 251.7201996
2: -169.7323761, 143.6659851, -167.7163086, 142.9577484, -312.6900940, 311.3822632
3: -83.4981003, 164.1896667, -83.0653000, 162.6521912, -246.1502991, 247.2549744
4: -184.1379089, 151.7879486, -181.9947968, 151.0111847, -335.1491089, 333.7827148

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.2303691, upper bound: 177.2182045
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0011005, upper bound: 176.0539322
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -146.9546051, 144.1370697, -107.5347595, 120.8612366, -267.8158264, 251.6718140
1: -121.2054749, 131.9143066, -88.9876251, 110.2532425, -231.4587097, 220.9019318
2: -169.7323761, 143.6659851, -124.8389893, 122.8079224, -292.5402527, 268.5049744
3: -83.4981003, 164.1896667, -72.6093903, 124.0585861, -207.5566711, 236.7990570
4: -184.1379089, 151.7879486, -135.6271362, 129.8474884, -313.9854126, 287.4150391

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.1655457, upper bound: 171.3814868
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.9713736, upper bound: 190.9982861
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -103.3768082, 118.4831009, -105.1731033, 120.4450989, -223.8218994, 223.6562042
1: -85.6280975, 107.8821106, -87.1250916, 109.8328781, -195.4609680, 195.0072021
2: -120.1185913, 120.5764313, -122.2827759, 122.3578644, -242.4764557, 242.8591919
3: -71.5149765, 119.5683746, -72.3044586, 121.5817719, -193.0967407, 191.8728333
4: -130.4989471, 127.4725266, -132.7809448, 129.3372955, -259.8362427, 260.2534790

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.3201656, upper bound: 171.2465594
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0025361, upper bound: 191.0012514
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -94.1150360, 113.0349884, -136.6329956, 142.3315582, -236.4465942, 249.6679840
1: -77.8260574, 102.4557419, -112.5237198, 130.7263031, -208.5523529, 214.9794617
2: -109.1581879, 115.4431381, -158.3572235, 142.7752991, -251.9334717, 273.8003540
3: -69.1613541, 109.2464905, -82.2227020, 155.9005585, -225.0619049, 191.4691925
4: -118.9238586, 122.2269440, -172.6639557, 150.3804169, -269.3042297, 294.8908386

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1399820, upper bound: 172.1167816
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.5040155, upper bound: 173.2895674
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0025194, upper bound: 191.0012538
time: 0.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.72 seconds
IS_A1_B1_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 4.72
Output dim: 0, lower bound: -168.2944111, upper bound: 170.0480897
IS_A1_B1_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 4.72
Output dim: 0, lower bound: -167.6340643, upper bound: 168.1179708
IS_A1_B1_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 4.72
Output dim: 0, lower bound: -172.6433144, upper bound: 174.9487920
IS_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -176.9238377, upper bound: 177.0821615
IS_A1_B1_B2_A1_A1, status: Status.VERIFIED, split count: 5, time: 4.72
Output dim: 0, lower bound: -171.2392459, upper bound: 172.3459127
IS_A1_B1_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 4.72
Output dim: 0, lower bound: -168.7579639, upper bound: 170.5839328
IS_A1_B1_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 4.72
Output dim: 0, lower bound: -171.2392459, upper bound: 172.3459127
IS_A1_B1_B2_A2_A2, status: Status.VERIFIED, split count: 5, time: 4.72
Output dim: 0, lower bound: -168.7579630, upper bound: 170.5839296
IS_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -176.3746705, upper bound: 176.3839162
IS_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -176.4478720, upper bound: 176.4004234
IS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -176.0754725, upper bound: 176.2556231
IS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -176.3287051, upper bound: 176.3287039
IS_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 4.72
Output dim: 0, lower bound: -172.7533011, upper bound: 171.1108861
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -177.9572734, upper bound: 180.3311065
IS_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -177.6099922, upper bound: 179.9655696
IS_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -177.4252643, upper bound: 179.8622274
IS_A2_B1_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 4.72
Output dim: 0, lower bound: -171.5075154, upper bound: 170.4498953
IS_A2_B1_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 4.72
Output dim: 0, lower bound: -170.9612000, upper bound: 169.6953763
IS_A2_B1_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 4.72
Output dim: 0, lower bound: -170.4883816, upper bound: 172.5825559
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -180.3681035, upper bound: 178.1595677
IS_A2_B1_B2_A1_A1, status: Status.VERIFIED, split count: 5, time: 4.72
Output dim: 0, lower bound: -170.6406446, upper bound: 172.4691610
IS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -180.3297872, upper bound: 177.9608545
IS_A2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 4.72
Output dim: 0, lower bound: -171.5040178, upper bound: 173.2895660
IS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -180.3297872, upper bound: 177.9570340
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -180.2303691, upper bound: 177.2182045
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.72
Output dim: 0, lower bound: -176.0011005, upper bound: 176.0539322
IS_A2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 4.72
Output dim: 0, lower bound: -174.1655457, upper bound: 171.3814868
IS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -190.9713736, upper bound: 190.9982861
IS_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 4.72
Output dim: 0, lower bound: -174.3201656, upper bound: 171.2465594
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -191.0025361, upper bound: 191.0012514
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.72
Output dim: 0, lower bound: -171.5040155, upper bound: 173.2895674
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -191.0025194, upper bound: 191.0012538

## BFS IS instance: IS_A1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -145.4437866, 155.1196899, -178.0234985, 173.9511871, -319.3949585, 333.1431580
1: -120.2910233, 142.7280731, -147.1725464, 160.3473358, -280.6383667, 289.9006348
2: -168.8097839, 153.7775879, -205.9979858, 171.7688446, -340.5786133, 359.7754822
3: -84.3281860, 166.2122345, -94.6885681, 199.3521423, -283.6803284, 260.9008179
4: -183.0110016, 162.3237000, -223.0717163, 181.1529694, -364.1639404, 385.3954163

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9078728, upper bound: 176.9639511
time: 0.68 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.7473274, upper bound: 176.0693546
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.6624663, upper bound: 175.6979422
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -130.0765381, 146.0296783, -130.2447510, 145.1062012, -275.1827393, 276.2744141
1: -107.9848175, 134.1199036, -108.3098297, 133.4948425, -241.4796600, 242.4297180
2: -151.5198669, 144.9234314, -151.9353485, 144.3334656, -295.8533020, 296.8587036
3: -80.0870819, 149.9904633, -79.6917114, 150.5039978, -230.5910645, 229.6821747
4: -163.8101654, 153.2248535, -163.8947754, 152.4680786, -316.2782593, 317.1196289

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.8963845, upper bound: 172.9958491
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -173.1663324, upper bound: 172.7746482
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -162.6228638, 169.2919312, -123.4955444, 140.8616943, -303.4844666, 292.7874451
1: -134.1692963, 156.3758850, -102.6653290, 129.5004883, -263.6697388, 259.0411987
2: -188.4640045, 168.7741241, -143.9890594, 140.2840881, -328.7480774, 312.7631531
3: -90.8654633, 185.0853119, -77.9645844, 142.9054565, -233.7709198, 263.0498962
4: -204.6768799, 177.7476654, -155.4329987, 148.3328247, -353.0097046, 333.1806641

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3932290, upper bound: 176.1755536
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3932290, upper bound: 176.4004234
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -134.7310486, 148.8347168, -132.6939697, 146.2593994, -280.9904175, 281.5286560
1: -111.5798416, 136.7331085, -110.1375961, 134.6177063, -246.1975403, 246.8706970
2: -156.5711975, 147.7487335, -154.4541321, 145.5255737, -302.0967407, 302.2028809
3: -81.6663971, 153.8104553, -80.1787109, 152.2830658, -233.9494629, 233.9891510
4: -169.6103821, 155.9773407, -166.9076691, 153.5224915, -323.1328735, 322.8850098

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -173.7334579, upper bound: 171.7383082
time: 0.73 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.7614182, upper bound: 175.9282252
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -162.6156769, 168.8672333, -125.9396210, 142.0432587, -304.6589355, 294.8068542
1: -134.1605682, 155.9385986, -104.4920197, 130.6379852, -264.7985535, 260.4306030
2: -188.5436707, 168.3849335, -146.5350189, 141.5045013, -330.0481567, 314.9199219
3: -91.0430145, 184.9252319, -78.4634399, 144.7287445, -235.7717590, 263.3886719
4: -204.8250427, 177.3227844, -158.4742737, 149.4298859, -354.2549438, 335.7970581

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2556243, upper bound: 176.0754714
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2556243, upper bound: 176.3287039
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -146.6312561, 157.2516022, -97.0911713, 114.2186813, -260.8499451, 254.3427734
1: -121.3761597, 144.7100830, -80.4074249, 103.9672852, -225.3434448, 225.1175079
2: -170.4106293, 155.9059601, -112.8006134, 116.4788971, -286.8895264, 268.7065735
3: -85.1553879, 167.9220581, -69.6611023, 112.8383636, -197.9937439, 237.5831299
4: -184.6484985, 164.5325928, -122.7283630, 123.1130066, -307.7614441, 287.2609253

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -172.5964573, upper bound: 174.6178984
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -172.5964573, upper bound: 180.3311075
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -164.1199646, 170.2151489, -84.3717880, 107.7749863, -271.8949585, 254.5869446
1: -135.3817139, 157.2136078, -69.9237213, 97.3087463, -232.6904602, 227.1373291
2: -190.2032928, 169.6698151, -98.0009460, 110.1542206, -300.3575134, 267.6707153
3: -91.2486725, 186.8899841, -66.7004242, 99.1340027, -190.3826752, 253.5904083
4: -206.6132812, 178.6828308, -106.7131042, 116.8855820, -323.4988098, 285.3959045

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -172.2201866, upper bound: 170.5012378
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.3010991, upper bound: 179.6904872
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -162.6472168, 168.8751221, -87.4559555, 108.9225006, -271.5697021, 256.3310242
1: -134.1835632, 155.9461365, -72.3239594, 98.4337311, -232.6172791, 228.2700806
2: -188.5745850, 168.3918610, -101.4120789, 111.3954849, -299.9700012, 269.8039551
3: -91.0465622, 184.9433441, -67.3129120, 101.5512543, -192.5978088, 252.2562561
4: -204.8613739, 177.3301239, -110.6191330, 118.0180206, -322.8793945, 287.9492493

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_A2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -172.5702815, upper bound: 174.7284891
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.1209787, upper bound: 179.5788422
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -103.2983704, 118.0250854, -182.9076538, 176.9971466, -280.2955322, 300.9327393
1: -85.5116882, 107.6285019, -150.8231964, 163.0276184, -248.5392609, 258.4516907
2: -119.9410553, 119.7767105, -211.2644501, 174.4967804, -294.4378357, 331.0411682
3: -71.1617203, 119.3753510, -95.7824173, 203.4107056, -274.5724182, 215.1577759
4: -130.3383026, 126.5624466, -229.0425720, 184.3425751, -314.6808777, 355.6049500

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.3559105, upper bound: 178.0549963
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.3202733, upper bound: 177.3250615
time: 0.58 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.4142177, upper bound: 172.4600403
time: 0.56 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -174.4267725, upper bound: 178.1595682
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -102.0467300, 118.2864304, -145.4621582, 155.3772888, -257.4239807, 263.7485962
1: -84.5236740, 107.8094711, -120.3638306, 142.9510803, -227.4747620, 228.1733093
2: -118.6239014, 119.9603577, -168.9502869, 153.9903564, -272.6142273, 288.9106445
3: -71.2450943, 118.1124725, -84.5951843, 166.4942322, -237.7393036, 202.7076569
4: -128.8798523, 126.7284317, -183.1327515, 162.5378876, -291.4177246, 309.8611755

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.0140589, upper bound: 172.9192778
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.0140589, upper bound: 177.9608550
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -131.8623962, 139.2340088, -136.3632660, 149.6460724, -281.5084839, 275.5972900
1: -108.5892868, 127.8554230, -112.8795471, 137.6061859, -246.1954651, 240.7349396
2: -152.8264465, 139.6357117, -158.4175110, 148.5879974, -301.4144287, 298.0531921
3: -80.6042938, 150.6765594, -81.8809204, 156.4909973, -237.0952911, 232.5574799
4: -166.7029572, 147.2871399, -171.7095337, 156.9573822, -323.6602783, 318.9966125

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.3498658, upper bound: 173.1567388
time: 0.58 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.3498658, upper bound: 177.9570345
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -140.6025085, 139.6843872, -143.0690613, 141.8341980, -282.4366455, 282.7534180
1: -115.9742508, 127.7212601, -118.0387497, 129.7845154, -245.7587585, 245.7599945
2: -162.3865051, 139.2944336, -165.2377930, 141.4601898, -303.8466797, 304.5322266
3: -81.5001144, 157.3161316, -82.3892212, 160.3209534, -241.8210754, 239.7053528
4: -176.2256470, 147.1510620, -179.3224945, 149.4248352, -325.6504211, 326.4735718

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2947071, upper bound: 174.2175276
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2902548, upper bound: 174.1986856
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -146.9546051, 144.1370697, -103.2983704, 118.0250854, -264.9796753, 247.4354401
1: -121.2054749, 131.9143066, -85.5116882, 107.6285019, -228.8339539, 217.4259796
2: -169.7323761, 143.6659851, -119.9410553, 119.7767105, -289.5090332, 263.6070557
3: -83.4981003, 164.1896667, -71.1617203, 119.3753510, -202.8734436, 235.3513794
4: -184.1379089, 151.7879486, -130.3383026, 126.5624466, -310.7003174, 282.1262512

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.9712550, upper bound: 190.9982854
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6333446, upper bound: 190.7118263
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6867623, upper bound: 190.7516968
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -103.3768082, 118.4831009, -102.0467300, 118.2864304, -221.6632385, 220.5298309
1: -85.6280975, 107.8821106, -84.5236740, 107.8094711, -193.4375610, 192.4057922
2: -120.1185913, 120.5764313, -118.6239014, 119.9603577, -240.0789490, 239.2003326
3: -71.5149765, 119.5683746, -71.2450943, 118.1124725, -189.6274414, 190.8134613
4: -130.4989471, 127.4725266, -128.8798523, 126.7284317, -257.2273865, 256.3523254

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.1108861, upper bound: 172.7533011
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -171.1108861, upper bound: 191.0012539
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -89.9584579, 110.0285873, -136.6329956, 142.3315582, -232.2900085, 246.6615448
1: -74.3320465, 99.8836517, -112.5237198, 130.7263031, -205.0583496, 212.4073639
2: -104.2582932, 112.5898285, -158.3572235, 142.7752991, -247.0335388, 270.9470520
3: -67.8981705, 104.6014099, -82.2227020, 155.9005585, -223.7987366, 186.8241119
4: -113.7484207, 119.1628494, -172.6639557, 150.3804169, -264.1288147, 291.8267517

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1399816, upper bound: 172.1167816
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1399820, upper bound: 172.1167817
time: 0.71 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.42 seconds
IS_A1_B1_B1_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.42
Output dim: 0, lower bound: -175.7473274, upper bound: 176.0693546
IS_A1_B1_B1_A2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.42
Output dim: 0, lower bound: -175.6624663, upper bound: 175.6979422
IS_A1_B2_B1_A1_A1_A1, status: Status.VERIFIED, split count: 6, time: 3.42
Output dim: 0, lower bound: -174.8963845, upper bound: 172.9958491
IS_A1_B2_B1_A1_A1_A2, status: Status.VERIFIED, split count: 6, time: 3.42
Output dim: 0, lower bound: -173.1663324, upper bound: 172.7746482
IS_A1_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -176.3932290, upper bound: 176.1755536
IS_A1_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -176.3932290, upper bound: 176.4004234
IS_A1_B2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.42
Output dim: 0, lower bound: -173.7334579, upper bound: 171.7383082
IS_A1_B2_B1_A2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.42
Output dim: 0, lower bound: -175.7614182, upper bound: 175.9282252
IS_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -176.2556243, upper bound: 176.0754714
IS_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -176.2556243, upper bound: 176.3287039
IS_A1_B2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 3.42
Output dim: 0, lower bound: -172.5964573, upper bound: 174.6178984
IS_A1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -172.5964573, upper bound: 180.3311075
IS_A1_B2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.42
Output dim: 0, lower bound: -172.2201866, upper bound: 170.5012378
IS_A1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -177.3010991, upper bound: 179.6904872
IS_A1_B2_B2_A2_A2_A1, status: Status.VERIFIED, split count: 6, time: 3.42
Output dim: 0, lower bound: -172.5702815, upper bound: 174.7284891
IS_A1_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -177.1209787, upper bound: 179.5788422
IS_A2_B1_B1_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.42
Output dim: 0, lower bound: -174.4142177, upper bound: 172.4600403
IS_A2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -174.4267725, upper bound: 178.1595682
IS_A2_B1_B2_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.42
Output dim: 0, lower bound: -175.0140589, upper bound: 172.9192778
IS_A2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -175.0140589, upper bound: 177.9608550
IS_A2_B1_B2_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.42
Output dim: 0, lower bound: -175.3498658, upper bound: 173.1567388
IS_A2_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -175.3498658, upper bound: 177.9570345
IS_A2_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -177.2947071, upper bound: 174.2175276
IS_A2_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -177.2902548, upper bound: 174.1986856
IS_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -190.6333446, upper bound: 190.7118263
IS_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -190.6867623, upper bound: 190.7516968
IS_A2_B2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 3.42
Output dim: 0, lower bound: -171.1108861, upper bound: 172.7533011
IS_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -171.1108861, upper bound: 191.0012539
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.42
Output dim: 0, lower bound: -175.1399816, upper bound: 172.1167816
IS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.42
Output dim: 0, lower bound: -175.1399820, upper bound: 172.1167817

## BFS IS instance: IS_A1_B2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -162.6228638, 169.2919312, -127.0132751, 143.3494110, -305.9722595, 296.3051453
1: -134.1692963, 156.3758850, -105.6452103, 131.8201294, -265.9894409, 262.0210876
2: -188.4640045, 168.7741241, -148.2217865, 142.5961609, -331.0601807, 316.9958191
3: -90.8654633, 185.0853119, -78.8576813, 147.1615906, -238.0270538, 263.9429932
4: -204.6768799, 177.7476654, -159.9141998, 150.6531219, -355.3299561, 337.6618652

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -172.2574150, upper bound: 174.0895958
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.7325923, upper bound: 172.7778596
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -162.6228638, 169.2919312, -154.8655853, 161.5279083, -324.1506958, 324.1575317
1: -134.1692963, 156.3758850, -128.1977692, 149.6977997, -283.8670959, 284.5736694
2: -188.4640045, 168.7741241, -180.0204010, 161.5238800, -349.9878540, 348.7945251
3: -90.8654633, 185.0853119, -87.3857727, 177.3312073, -268.1966553, 272.4710693
4: -204.6768799, 177.7476654, -194.9322510, 169.9468384, -374.6237183, 372.6799316

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -169.5335041, upper bound: 171.0842529
time: 0.73 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0633310, upper bound: 175.9715548
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -162.6156769, 168.8672333, -129.0959778, 144.3085327, -306.9241943, 297.9631958
1: -134.1605682, 155.9385986, -107.1895752, 132.7637634, -266.9243164, 263.1281738
2: -188.5436707, 168.3849335, -150.3433380, 143.5924835, -332.1361694, 318.7282715
3: -91.0430145, 184.9252319, -79.2600937, 148.5979004, -239.6409149, 264.1853333
4: -204.8250427, 177.3227844, -162.4703827, 151.5149994, -356.3400269, 339.7931519

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.4149896, upper bound: 175.5172822
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.3368819, upper bound: 175.3251875
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -162.6156769, 168.8672333, -154.3970947, 161.1812439, -323.7969360, 323.2643433
1: -134.1605682, 155.9385986, -127.8102875, 149.3794403, -283.5400085, 283.7489014
2: -188.5436707, 168.3849335, -179.4863434, 161.1655426, -349.7092285, 347.8712463
3: -91.0430145, 184.9252319, -87.2749634, 176.6403503, -267.6833496, 272.2001648
4: -204.8250427, 177.3227844, -194.3441467, 169.5900116, -374.4150085, 371.6669312

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -169.4290081, upper bound: 172.5260844
time: 0.72 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9282265, upper bound: 175.9715548
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -142.6867676, 154.5982056, -97.0911713, 114.2186813, -256.9054260, 251.6893768
1: -118.1423569, 142.2582245, -80.4074249, 103.9672852, -222.1096344, 222.6656494
2: -165.8412476, 153.3789673, -112.8006134, 116.4788971, -282.3200989, 266.1795654
3: -83.6975327, 163.4712982, -69.6611023, 112.8383636, -196.5358887, 233.1324005
4: -179.6697388, 161.8748322, -122.7283630, 123.1130066, -302.7826233, 284.6032104

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -170.6409050, upper bound: 179.1383155
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -170.6409050, upper bound: 179.2119852
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -164.1199646, 170.2151489, -80.7284698, 105.1127319, -269.2326965, 250.9436188
1: -135.3817139, 157.2136078, -66.8516693, 94.8672409, -230.2489471, 224.0652771
2: -190.2032928, 169.6698151, -93.6992950, 107.6341095, -297.8374023, 263.3690491
3: -91.2486725, 186.8899841, -65.5953369, 95.0271530, -186.2758179, 252.4853058
4: -206.6132812, 178.6828308, -102.1688690, 114.1716156, -320.7849121, 280.8516235

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -172.1621934, upper bound: 174.1377863
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -172.1621934, upper bound: 179.6904844
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -158.4542542, 166.1195068, -87.4559555, 108.9225006, -267.3767700, 253.5754700
1: -130.7510529, 153.3862000, -72.3239594, 98.4337311, -229.1847229, 225.7101593
2: -183.7286530, 165.7667236, -101.4120789, 111.3954849, -295.1241150, 267.1788025
3: -89.5624695, 180.2417908, -67.3129120, 101.5512543, -191.1137238, 247.5547028
4: -199.5887146, 174.5928802, -110.6191330, 118.0180206, -317.6067505, 285.2119751

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B2_A2_A2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2498289, upper bound: 178.8605913
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.3497818, upper bound: 175.6399288
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -103.2983704, 118.0250854, -178.0838165, 173.9530029, -277.2513733, 296.1088867
1: -85.5116882, 107.6285019, -146.8449402, 160.1979523, -245.7096405, 254.4734039
2: -119.9410553, 119.7767105, -205.6577759, 171.5931854, -291.5342407, 325.4344788
3: -71.1617203, 119.3753510, -94.0453568, 198.0908966, -269.2526245, 213.4207153
4: -130.3383026, 126.5624466, -222.9821625, 181.2675629, -311.6058655, 349.5445862

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -174.4267725, upper bound: 178.0549968
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -171.5299390, upper bound: 177.3250622
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.1304182, upper bound: 170.0606995
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -169.5651258, upper bound: 168.0510196
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -102.0467300, 118.2864304, -141.1893005, 152.5228729, -254.5695648, 259.4757080
1: -84.5236740, 107.8094711, -116.8425293, 140.3120422, -224.8357239, 224.6520081
2: -118.6239014, 119.9603577, -163.9844971, 151.2703094, -269.8942261, 283.9448547
3: -71.2450943, 118.1124725, -83.0364075, 161.6668701, -232.9119568, 201.1488647
4: -128.8798523, 126.7284317, -177.7456055, 159.6779633, -288.5578003, 304.4740295

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -170.8907472, upper bound: 169.9484928
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -174.6178988, upper bound: 177.9608550
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -174.6178988, upper bound: 177.9608327
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -131.8623962, 139.2340088, -131.9849854, 146.7486877, -278.6110840, 271.2189941
1: -108.5892868, 127.8554230, -109.2580261, 134.9039001, -243.4931641, 237.1134491
2: -152.8264465, 139.6357117, -153.3062592, 145.8266754, -298.6531372, 292.9419250
3: -80.6042938, 150.6765594, -80.3545685, 151.4954681, -232.0997467, 231.0311279
4: -166.7029572, 147.2871399, -166.1802979, 154.0709381, -320.7738342, 313.4672852

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -174.6178988, upper bound: 177.9570345
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -174.6178988, upper bound: 177.9570123
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -127.0661469, 131.2058105, -133.7399597, 135.8167572, -262.8829041, 264.9457703
1: -104.8562698, 119.7991257, -110.3474579, 124.0946198, -228.9508972, 230.1465759
2: -146.8104858, 132.0823364, -154.4355011, 136.3356781, -283.1461792, 286.5178223
3: -77.8504486, 143.3646240, -79.9632416, 150.3858948, -228.2363281, 223.3278656
4: -159.5704956, 139.6388397, -167.8778992, 144.1257019, -303.6961365, 307.5167236

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B1_A1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.9541627, upper bound: 173.5284368
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.0501507, upper bound: 172.5034229
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -128.6567841, 130.6178436, -136.9611359, 137.0615387, -265.7183228, 267.5789490
1: -106.2074432, 119.1298065, -113.0435104, 125.2615433, -231.4689789, 232.1733093
2: -148.5057831, 131.4829712, -158.1432495, 137.3147736, -285.8205566, 289.6262207
3: -77.9452896, 143.0504456, -80.5288010, 152.8164673, -230.7617493, 223.5792542
4: -161.0715637, 139.0860291, -171.5577850, 145.1475220, -306.2190857, 310.6437683

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.9614628, upper bound: 169.0440925
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.5051358, upper bound: 173.4412958
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -133.1066284, 135.4988861, -95.0899277, 113.0487747, -246.1553802, 230.5888062
1: -109.8398285, 123.8449860, -78.9085388, 102.4782028, -212.3180237, 202.7535248
2: -153.8077698, 136.2717438, -110.6227722, 115.2914505, -269.0992126, 246.8945160
3: -79.7512207, 149.9747467, -69.1612473, 110.6598969, -190.4111176, 219.1359863
4: -167.0955200, 144.0845795, -120.0587616, 122.0549469, -289.1504517, 264.1433411

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.4839026, upper bound: 190.6238096
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.4841918, upper bound: 190.4545261
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -134.9871216, 135.0418396, -97.7953568, 114.0565643, -249.0436249, 232.8371887
1: -111.4261322, 123.2940369, -80.9698944, 103.8179092, -215.2440491, 204.2639313
2: -155.8353271, 135.7733612, -113.5206757, 116.3823013, -272.2175903, 249.2940369
3: -79.9042892, 149.9246674, -69.6832428, 112.7467117, -192.6510010, 219.6079102
4: -168.9441986, 143.6371460, -123.4323273, 123.0314331, -291.9756470, 267.0694580

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6867622, upper bound: 190.7516960
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.5833184, upper bound: 190.7019641
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -170.0911064, upper bound: 190.6426027
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -99.9376678, 115.9718246, -102.0467300, 118.2864304, -218.2240906, 218.0185547
1: -82.7733231, 105.6485443, -84.5236740, 107.8094711, -190.5827942, 190.1722107
2: -116.1020203, 117.9590607, -118.6239014, 119.9603577, -236.0623779, 236.5829620
3: -70.3483124, 115.7176437, -71.2450943, 118.1124725, -188.4607849, 186.9627228
4: -126.2090149, 124.6291351, -128.8798523, 126.7284317, -252.9374390, 253.5089874

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -169.8992475, upper bound: 190.9712493
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -169.8992501, upper bound: 169.9073608
time: 0.77 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.64 seconds
IS_A1_B2_B1_A1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.64
Output dim: 0, lower bound: -172.2574150, upper bound: 174.0895958
IS_A1_B2_B1_A1_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.64
Output dim: 0, lower bound: -171.7325923, upper bound: 172.7778596
IS_A1_B2_B1_A1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.64
Output dim: 0, lower bound: -169.5335041, upper bound: 171.0842529
IS_A1_B2_B1_A1_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 4.64
Output dim: 0, lower bound: -176.0633310, upper bound: 175.9715548
IS_A1_B2_B1_A2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.64
Output dim: 0, lower bound: -175.4149896, upper bound: 175.5172822
IS_A1_B2_B1_A2_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.64
Output dim: 0, lower bound: -175.3368819, upper bound: 175.3251875
IS_A1_B2_B1_A2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.64
Output dim: 0, lower bound: -169.4290081, upper bound: 172.5260844
IS_A1_B2_B1_A2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 4.64
Output dim: 0, lower bound: -175.9282265, upper bound: 175.9715548
IS_A1_B2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.64
Output dim: 0, lower bound: -170.6409050, upper bound: 179.1383155
IS_A1_B2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.64
Output dim: 0, lower bound: -170.6409050, upper bound: 179.2119852
IS_A1_B2_B2_A2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.64
Output dim: 0, lower bound: -172.1621934, upper bound: 174.1377863
IS_A1_B2_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.64
Output dim: 0, lower bound: -172.1621934, upper bound: 179.6904844
IS_A1_B2_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.64
Output dim: 0, lower bound: -176.2498289, upper bound: 178.8605913
IS_A1_B2_B2_A2_A2_A2_B2, status: Status.VERIFIED, split count: 7, time: 4.64
Output dim: 0, lower bound: -175.3497818, upper bound: 175.6399288
IS_A2_B1_B1_A2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.64
Output dim: 0, lower bound: -171.1304182, upper bound: 170.0606995
IS_A2_B1_B1_A2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 4.64
Output dim: 0, lower bound: -169.5651258, upper bound: 168.0510196
IS_A2_B1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.64
Output dim: 0, lower bound: -174.6178988, upper bound: 177.9608550
IS_A2_B1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.64
Output dim: 0, lower bound: -174.6178988, upper bound: 177.9608327
IS_A2_B1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.64
Output dim: 0, lower bound: -174.6178988, upper bound: 177.9570345
IS_A2_B1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.64
Output dim: 0, lower bound: -174.6178988, upper bound: 177.9570123
IS_A2_B2_A1_B1_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 4.64
Output dim: 0, lower bound: -174.9541627, upper bound: 173.5284368
IS_A2_B2_A1_B1_A1_A1_B2, status: Status.VERIFIED, split count: 7, time: 4.64
Output dim: 0, lower bound: -175.0501507, upper bound: 172.5034229
IS_A2_B2_A1_B1_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 4.64
Output dim: 0, lower bound: -171.9614628, upper bound: 169.0440925
IS_A2_B2_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.64
Output dim: 0, lower bound: -176.5051358, upper bound: 173.4412958
IS_A2_B2_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.64
Output dim: 0, lower bound: -190.4839026, upper bound: 190.6238096
IS_A2_B2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.64
Output dim: 0, lower bound: -190.4841918, upper bound: 190.4545261
IS_A2_B2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.64
Output dim: 0, lower bound: -190.5833184, upper bound: 190.7019641
IS_A2_B2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.64
Output dim: 0, lower bound: -170.0911064, upper bound: 190.6426027
IS_A2_B2_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.64
Output dim: 0, lower bound: -169.8992475, upper bound: 190.9712493
IS_A2_B2_A2_B1_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 4.64
Output dim: 0, lower bound: -169.8992501, upper bound: 169.9073608

## BFS IS instance: IS_A1_B2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -142.6867676, 154.5982056, -93.0543518, 112.1555939, -254.8423157, 247.6525574
1: -118.1423569, 142.2582245, -77.0974884, 101.8335037, -219.9758148, 219.3557129
2: -165.8412476, 153.3789673, -108.1632690, 114.3950806, -280.2362976, 261.5421753
3: -83.6975327, 163.4712982, -68.6649551, 108.6678238, -192.3653564, 232.1362305
4: -179.6697388, 161.8748322, -117.7151260, 120.9675598, -300.6372681, 279.5899353

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6502169, upper bound: 178.3416536
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3758882, upper bound: 178.0206364
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -142.6867676, 154.5982056, -116.6109695, 126.9670258, -269.6537170, 271.2091675
1: -118.1423569, 142.2582245, -95.9963150, 116.6380920, -234.7804260, 238.2545471
2: -165.8412476, 153.3789673, -135.0545349, 128.9477539, -294.7889709, 288.4335022
3: -83.6975327, 163.4712982, -75.7488632, 133.8908386, -217.5883789, 239.2201538
4: -179.6697388, 161.8748322, -147.4843903, 135.8696442, -315.5393677, 309.3592224

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6502169, upper bound: 178.5025453
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3758871, upper bound: 178.3994876
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -159.8971100, 167.4477539, -80.7284698, 105.1127319, -265.0098267, 248.1761780
1: -131.9362183, 154.6526489, -66.8516693, 94.8672409, -226.8034668, 221.5043182
2: -185.3460083, 167.0395508, -93.6992950, 107.6341095, -292.9801025, 260.7388000
3: -89.7636490, 182.1958313, -65.5953369, 95.0271530, -184.7908020, 247.7911530
4: -201.3185883, 175.9373474, -102.1688690, 114.1716156, -315.4902039, 278.1062012

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -168.0484641, upper bound: 176.5024354
time: 0.77 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -168.0484641, upper bound: 176.5997122
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -156.7115326, 164.8421631, -83.5053940, 105.9911118, -262.7026367, 248.3475647
1: -129.3176117, 152.1855927, -69.0429535, 95.5938187, -224.9114075, 221.2285461
2: -181.7102509, 164.5257568, -96.7925415, 108.5142059, -290.2244568, 261.3182678
3: -89.0089111, 178.3114014, -66.1056137, 97.1786499, -186.1875610, 244.4170227
4: -197.4080505, 173.2862244, -105.6866379, 114.9753113, -312.3833618, 278.9728699

Time for backsubstitution: 1.95 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=279.9653625488281
rel_dist={0: [-191.43286165766193, 191.43286165766187]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1128.10 seconds
