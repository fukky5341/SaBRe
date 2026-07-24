## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_5.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 554.967677004936


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141)
1: (-208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797)
2: (-176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039)
3: (-185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579)
4: (-158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706)

## BASE Result
execution time: IAR + LP analysis = 2.33 + 2.20 = 4.54 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -554.9912953, upper bound: 554.9912953


# Binary Search by BASE starts (time budget: 1195.46 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=619.6494140625
rel_dist={0: [-554.9911089992902, 554.9911089992902]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=619.6494140625
rel_dist={0: [-554.9907236424904, 554.9907236424906]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=619.6494140625
rel_dist={0: [-554.990153595014, 554.9901535950139]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=619.6494140625
rel_dist={0: [-554.988966335711, 554.988966335711]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=619.6494140625
rel_dist={0: [-554.9880975414051, 554.9880975414051]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=619.6494140625
rel_dist={0: [-554.9876280275749, 554.9876280275748]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=619.6494140625
rel_dist={0: [-554.9873672980016, 554.9873672980016]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=619.6494140625
rel_dist={0: [-554.9872352391235, 554.9872352391233]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=619.6494140625
rel_dist={0: [-554.9871688329926, 554.9871688329922]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=619.6494140625
rel_dist={0: [-554.9871355495204, 554.9871355495204]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=619.6494140625
rel_dist={0: [-554.9871181509133, 554.9871181509134]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=619.6494140625
rel_dist={0: [-554.9871092291382, 554.9871092291382]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=619.6494140625
rel_dist={0: [-554.9871047682793, 554.9871047682793]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=619.6494140625
rel_dist={0: [-554.9871025379067, 554.9871025379066]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=619.6494140625
rel_dist={0: [-554.9871014240496, 554.9871014228313]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=619.6494140625
rel_dist={0: [-554.9871008661073, 554.9871008655066]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=619.6494140625
rel_dist={0: [-554.9871005875302, 554.9871006005683]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=619.6494140625
rel_dist={0: [-554.9871004489975, 554.9871004644856]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=619.6494140625
rel_dist={0: [-554.9871004039769, 554.9871004148458]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=619.6494140625
rel_dist={0: [-554.987100444383, 554.9871004366394]}

## Binary Search Result
Binary search time: 93.11 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1102.36 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9846656
time: 1.03 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9794534, upper bound: 554.9794534
time: 1.31 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.51 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.51
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9846656
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.51
Output dim: 0, lower bound: -554.9794534, upper bound: 554.9794534

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -135.4273682, 427.6317749, -148.0532990, 471.0341492, -606.4615479, 575.6850586
1: -189.9990234, 430.8988953, -208.2655029, 473.8415222, -663.8405762, 639.1641846
2: -160.4833527, 476.7970886, -175.8965302, 523.9502563, -684.4335938, 652.6936035
3: -169.3506470, 612.0695801, -185.5812531, 673.2926025, -842.6431274, 797.6507568
4: -144.2209015, 559.9951782, -158.0018616, 615.2876587, -759.5085449, 717.9970093

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9794534, upper bound: 554.9794534
time: 1.05 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9794534, upper bound: 554.9794534
time: 1.48 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -141.9587555, 452.3752747, -148.1843414, 471.4650574, -613.4238281, 600.5596313
1: -199.8379517, 454.9771729, -208.4494476, 474.2722168, -674.1101074, 663.4266357
2: -168.7648926, 503.0252075, -176.0516663, 524.4273071, -693.1921387, 679.0769043
3: -178.0149994, 646.6646729, -185.7463837, 673.9080200, -851.9229736, 832.4110718
4: -151.5778503, 590.7100830, -158.1413727, 615.8510742, -767.4288940, 748.8513794

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9794534, upper bound: 554.9794534
time: 0.98 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9794534, upper bound: 554.9794534
time: 1.43 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.45 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.45
Output dim: 0, lower bound: -554.9794534, upper bound: 554.9794534
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.45
Output dim: 0, lower bound: -554.9794534, upper bound: 554.9794534
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.45
Output dim: 0, lower bound: -554.9794534, upper bound: 554.9794534
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.45
Output dim: 0, lower bound: -554.9794534, upper bound: 554.9794534

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -135.4273682, 427.6317749, -135.4273682, 427.6317749, -563.0591431, 563.0591431
1: -189.9990234, 430.8988953, -189.9990234, 430.8988953, -620.8978271, 620.8978271
2: -160.4833527, 476.7970886, -160.4833527, 476.7970886, -637.2804565, 637.2804565
3: -169.3506470, 612.0695801, -169.3506470, 612.0695801, -781.4201660, 781.4201050
4: -144.2209015, 559.9951782, -144.2209015, 559.9951782, -704.2160645, 704.2160645

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9846656
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9836280
time: 1.19 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -135.4273682, 427.6317749, -141.9587555, 452.3752747, -587.8026123, 569.5904541
1: -189.9990234, 430.8988953, -199.8379517, 454.9771729, -644.9761963, 630.7367554
2: -160.4833527, 476.7970886, -168.7648926, 503.0252075, -663.5085449, 645.5618896
3: -169.3506470, 612.0695801, -178.0149994, 646.6646729, -816.0152588, 790.0845947
4: -144.2209015, 559.9951782, -151.5778503, 590.7100830, -734.9309692, 711.5729980

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9846656
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9836280
time: 1.34 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -141.9587555, 452.3752747, -135.4273682, 427.6317749, -569.5905151, 587.8026123
1: -199.8379517, 454.9771729, -189.9990234, 430.8988953, -630.7367554, 644.9761963
2: -168.7648926, 503.0252075, -160.4833527, 476.7970886, -645.5618896, 663.5085449
3: -178.0149994, 646.6646729, -169.3506470, 612.0695801, -790.0845337, 816.0152588
4: -151.5778503, 590.7100830, -144.2209015, 559.9951782, -711.5729980, 734.9309692

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9789852, upper bound: 554.9794534
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9794534, upper bound: 554.9794534
time: 1.37 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -141.9587555, 452.3752747, -141.9587555, 452.3752747, -594.3340454, 594.3339844
1: -199.8379517, 454.9771729, -199.8379517, 454.9771729, -654.8151245, 654.8151245
2: -168.7648926, 503.0252075, -168.7648926, 503.0252075, -671.7901001, 671.7901001
3: -178.0149994, 646.6646729, -178.0149994, 646.6646729, -824.6796875, 824.6796875
4: -151.5778503, 590.7100830, -151.5778503, 590.7100830, -742.2879028, 742.2879028

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9789852, upper bound: 554.9794534
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9794534, upper bound: 554.9794534
time: 1.29 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.68 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.68
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9846656
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.68
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9836280
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.68
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9846656
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.68
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9836280
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.68
Output dim: 0, lower bound: -554.9789852, upper bound: 554.9794534
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.68
Output dim: 0, lower bound: -554.9794534, upper bound: 554.9794534
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.68
Output dim: 0, lower bound: -554.9789852, upper bound: 554.9794534
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.68
Output dim: 0, lower bound: -554.9794534, upper bound: 554.9794534

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -115.3944626, 363.5161438, -135.4273682, 427.6317749, -543.0262451, 498.9435120
1: -161.7974091, 366.7948303, -189.9990234, 430.8988953, -592.6961060, 556.7938232
2: -136.8181305, 405.7499084, -160.4833527, 476.7970886, -613.6152344, 566.2332764
3: -144.2350464, 521.4130859, -169.3506470, 612.0695801, -756.3046265, 690.7636719
4: -123.1167984, 476.9207153, -144.2209015, 559.9951782, -683.1119385, 621.1416016

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9898618, upper bound: 554.9898618
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9898618, upper bound: 554.9898618
time: 1.28 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -131.5360107, 414.6815186, -135.4273682, 427.6317749, -559.1677856, 550.1088867
1: -184.4300995, 418.1284180, -189.9990234, 430.8988953, -615.3288574, 608.1274414
2: -155.8180084, 462.7027588, -160.4833527, 476.7970886, -632.6150513, 623.1860962
3: -164.3947754, 593.6577148, -169.3506470, 612.0695801, -776.4642944, 763.0082397
4: -140.0432434, 543.4032593, -144.2209015, 559.9951782, -700.0384521, 687.6241455

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9898618, upper bound: 554.9898618
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9898618, upper bound: 554.9898618
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -115.3944626, 363.5161438, -141.9587555, 452.3752747, -567.7697144, 505.4749146
1: -161.7974091, 366.7948303, -199.8379517, 454.9771729, -616.7745972, 566.6328125
2: -136.8181305, 405.7499084, -168.7648926, 503.0252075, -639.8433228, 574.5147705
3: -144.2350464, 521.4130859, -178.0149994, 646.6646729, -790.8997192, 699.4281006
4: -123.1167984, 476.9207153, -151.5778503, 590.7100830, -713.8268433, 628.4985352

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9831599
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9836280
time: 1.17 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -131.5360107, 414.6815186, -141.9587555, 452.3752747, -583.9112549, 556.6402588
1: -184.4300995, 418.1284180, -199.8379517, 454.9771729, -639.4072876, 617.9663696
2: -155.8180084, 462.7027588, -168.7648926, 503.0252075, -658.8432007, 631.4675903
3: -164.3947754, 593.6577148, -178.0149994, 646.6646729, -811.0594482, 771.6726685
4: -140.0432434, 543.4032593, -151.5778503, 590.7100830, -730.7532959, 694.9810791

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9831599
time: 1.38 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9836280
time: 1.56 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -123.2541962, 392.5810242, -135.4273682, 427.6317749, -550.8859863, 528.0084229
1: -173.5091553, 394.9635315, -189.9990234, 430.8988953, -604.4079590, 584.9625244
2: -146.6661682, 436.5891418, -160.4833527, 476.7970886, -623.4630737, 597.0724487
3: -154.5595398, 561.9185181, -169.3506470, 612.0695801, -766.6290894, 731.2690430
4: -131.8280182, 512.9211426, -144.2209015, 559.9951782, -691.8231812, 657.1420288

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9831599, upper bound: 554.9856871
time: 1.36 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9831599, upper bound: 554.9856871
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -137.6115265, 437.8572693, -135.4273682, 427.6317749, -565.2432861, 573.2846680
1: -193.6290741, 440.6151428, -189.9990234, 430.8988953, -624.5277710, 630.6141357
2: -163.5594940, 487.1738281, -160.4833527, 476.7970886, -640.3565674, 647.6571655
3: -172.4905701, 626.0422974, -169.3506470, 612.0695801, -784.5599976, 795.3928833
4: -146.9165649, 572.0592651, -144.2209015, 559.9951782, -706.9117432, 716.2801514

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9836280, upper bound: 554.9856871
time: 1.08 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9836280, upper bound: 554.9856871
time: 1.20 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -123.2541962, 392.5810242, -141.9587555, 452.3752747, -575.6294556, 534.5397949
1: -173.5091553, 394.9635315, -199.8379517, 454.9771729, -628.4863281, 594.8015137
2: -146.6661682, 436.5891418, -168.7648926, 503.0252075, -649.6912231, 605.3538818
3: -154.5595398, 561.9185181, -178.0149994, 646.6646729, -801.2241821, 739.9334106
4: -131.8280182, 512.9211426, -151.5778503, 590.7100830, -722.5380859, 664.4989624

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9789852, upper bound: 554.9789852
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9789852, upper bound: 554.9794534
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -137.6115265, 437.8572693, -141.9587555, 452.3752747, -589.9868164, 579.8160400
1: -193.6290741, 440.6151428, -199.8379517, 454.9771729, -648.6062622, 640.4531250
2: -163.5594940, 487.1738281, -168.7648926, 503.0252075, -666.5847168, 655.9386597
3: -172.4905701, 626.0422974, -178.0149994, 646.6646729, -819.1551514, 804.0573120
4: -146.9165649, 572.0592651, -151.5778503, 590.7100830, -737.6265869, 723.6370850

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9794534, upper bound: 554.9789852
time: 1.28 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9794534, upper bound: 554.9794534
time: 1.22 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.54 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -554.9898618, upper bound: 554.9898618
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -554.9898618, upper bound: 554.9898618
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -554.9898618, upper bound: 554.9898618
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -554.9898618, upper bound: 554.9898618
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9831599
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9836280
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9831599
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9836280
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -554.9831599, upper bound: 554.9856871
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -554.9831599, upper bound: 554.9856871
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -554.9836280, upper bound: 554.9856871
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -554.9836280, upper bound: 554.9856871
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -554.9789852, upper bound: 554.9789852
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -554.9789852, upper bound: 554.9794534
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -554.9794534, upper bound: 554.9789852
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -554.9794534, upper bound: 554.9794534

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -115.3944626, 363.5161438, -115.3944626, 363.5161438, -478.9105835, 478.9105835
1: -161.7974091, 366.7948303, -161.7974091, 366.7948303, -528.5922241, 528.5922241
2: -136.8181305, 405.7499084, -136.8181305, 405.7499084, -542.5680542, 542.5680542
3: -144.2350464, 521.4130859, -144.2350464, 521.4130859, -665.6481323, 665.6481323
4: -123.1167984, 476.9207153, -123.1167984, 476.9207153, -600.0375366, 600.0375366

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9861204, upper bound: 554.9861084
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9874099, upper bound: 554.9893278
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -115.3944626, 363.5161438, -131.5360107, 414.6815186, -530.0759888, 495.0521545
1: -161.7974091, 366.7948303, -184.4300995, 418.1284180, -579.9256592, 551.2249146
2: -136.8181305, 405.7499084, -155.8180084, 462.7027588, -599.5208740, 561.5679321
3: -144.2350464, 521.4130859, -164.3947754, 593.6577148, -737.8927002, 685.8078613
4: -123.1167984, 476.9207153, -140.0432434, 543.4032593, -666.5200806, 616.9639893

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9861204, upper bound: 554.9861084
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9874099, upper bound: 554.9893278
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -131.5360107, 414.6815186, -115.3944626, 363.5161438, -495.0521545, 530.0759888
1: -184.4300995, 418.1284180, -161.7974091, 366.7948303, -551.2249146, 579.9256592
2: -155.8180084, 462.7027588, -136.8181305, 405.7499084, -561.5679321, 599.5208740
3: -164.3947754, 593.6577148, -144.2350464, 521.4130859, -685.8078613, 737.8927002
4: -140.0432434, 543.4032593, -123.1167984, 476.9207153, -616.9639893, 666.5200806

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9863208, upper bound: 554.9862737
time: 1.33 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9873959, upper bound: 554.9873959
time: 1.25 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -131.5360107, 414.6815186, -131.5360107, 414.6815186, -546.2175293, 546.2175293
1: -184.4300995, 418.1284180, -184.4300995, 418.1284180, -602.5584106, 602.5584717
2: -155.8180084, 462.7027588, -155.8180084, 462.7027588, -618.5207520, 618.5207520
3: -164.3947754, 593.6577148, -164.3947754, 593.6577148, -758.0524292, 758.0524292
4: -140.0432434, 543.4032593, -140.0432434, 543.4032593, -683.4465332, 683.4465332

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9863208, upper bound: 554.9862737
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9873959, upper bound: 554.9873959
time: 1.40 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -115.3944626, 363.5161438, -123.2541962, 392.5810242, -507.9754639, 486.7703247
1: -161.7974091, 366.7948303, -173.5091553, 394.9635315, -556.7608643, 540.3039551
2: -136.8181305, 405.7499084, -146.6661682, 436.5891418, -573.4072266, 552.4159546
3: -144.2350464, 521.4130859, -154.5595398, 561.9185181, -706.1535034, 675.9725952
4: -123.1167984, 476.9207153, -131.8280182, 512.9211426, -636.0379028, 608.7487183

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9842999
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855172, upper bound: 554.9828575
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -115.3944626, 363.5161438, -137.6115265, 437.8572693, -553.2517090, 501.1276550
1: -161.7974091, 366.7948303, -193.6290741, 440.6151428, -602.4124146, 560.4238892
2: -136.8181305, 405.7499084, -163.5594940, 487.1738281, -623.9919434, 569.3093872
3: -144.2350464, 521.4130859, -172.4905701, 626.0422974, -770.2773438, 693.9035034
4: -123.1167984, 476.9207153, -146.9165649, 572.0592651, -695.1760864, 623.8372803

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9846656
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855172, upper bound: 554.9833256
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -131.5360107, 414.6815186, -123.2541962, 392.5810242, -524.1170654, 537.9357300
1: -184.4300995, 418.1284180, -173.5091553, 394.9635315, -579.3936157, 591.6375732
2: -155.8180084, 462.7027588, -146.6661682, 436.5891418, -592.4070435, 609.3687744
3: -164.3947754, 593.6577148, -154.5595398, 561.9185181, -726.3132324, 748.2171631
4: -140.0432434, 543.4032593, -131.8280182, 512.9211426, -652.9643555, 675.2312622

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9831599
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855172, upper bound: 554.9827202
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -131.5360107, 414.6815186, -137.6115265, 437.8572693, -569.3933105, 552.2930298
1: -184.4300995, 418.1284180, -193.6290741, 440.6151428, -625.0452271, 611.7573853
2: -155.8180084, 462.7027588, -163.5594940, 487.1738281, -642.9918213, 626.2622681
3: -164.3947754, 593.6577148, -172.4905701, 626.0422974, -790.4370728, 766.1481323
4: -140.0432434, 543.4032593, -146.9165649, 572.0592651, -712.1025391, 690.3198242

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9831924
time: 1.30 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855172, upper bound: 554.9827752
time: 1.24 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -123.2541962, 392.5810242, -115.3944626, 363.5161438, -486.7703247, 507.9754639
1: -173.5091553, 394.9635315, -161.7974091, 366.7948303, -540.3039551, 556.7608643
2: -146.6661682, 436.5891418, -136.8181305, 405.7499084, -552.4159546, 573.4072266
3: -154.5595398, 561.9185181, -144.2350464, 521.4130859, -675.9725952, 706.1535034
4: -131.8280182, 512.9211426, -123.1167984, 476.9207153, -608.7487183, 636.0379028

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9782383, upper bound: 554.9825392
time: 1.22 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9570784, upper bound: 554.9776358
time: 1.25 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -123.2541962, 392.5810242, -131.5360107, 414.6815186, -537.9357300, 524.1170654
1: -173.5091553, 394.9635315, -184.4300995, 418.1284180, -591.6375122, 579.3936157
2: -146.6661682, 436.5891418, -155.8180084, 462.7027588, -609.3687744, 592.4070435
3: -154.5595398, 561.9185181, -164.3947754, 593.6577148, -748.2171631, 726.3132324
4: -131.8280182, 512.9211426, -140.0432434, 543.4032593, -675.2312622, 652.9643555

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9782383, upper bound: 554.9825392
time: 1.24 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9570784, upper bound: 554.9776358
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -137.6115265, 437.8572693, -115.3944626, 363.5161438, -501.1276550, 553.2517090
1: -193.6290741, 440.6151428, -161.7974091, 366.7948303, -560.4238892, 602.4124146
2: -163.5594940, 487.1738281, -136.8181305, 405.7499084, -569.3093872, 623.9919434
3: -172.4905701, 626.0422974, -144.2350464, 521.4130859, -693.9035034, 770.2773438
4: -146.9165649, 572.0592651, -123.1167984, 476.9207153, -623.8372803, 695.1760864

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9786397, upper bound: 554.9825392
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9628015, upper bound: 554.9797231
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9834286, upper bound: 554.9853005
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -137.6115265, 437.8572693, -131.5360107, 414.6815186, -552.2930298, 569.3933105
1: -193.6290741, 440.6151428, -184.4300995, 418.1284180, -611.7573242, 625.0452271
2: -163.5594940, 487.1738281, -155.8180084, 462.7027588, -626.2622681, 642.9918213
3: -172.4905701, 626.0422974, -164.3947754, 593.6577148, -766.1481323, 790.4370728
4: -146.9165649, 572.0592651, -140.0432434, 543.4032593, -690.3198242, 712.1025391

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9786397, upper bound: 554.9825392
time: 1.27 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9628015, upper bound: 554.9797231
time: 1.51 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9834286, upper bound: 554.9853005
time: 1.30 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -123.2541962, 392.5810242, -123.2541962, 392.5810242, -515.8352051, 515.8352051
1: -173.5091553, 394.9635315, -173.5091553, 394.9635315, -568.4726562, 568.4726562
2: -146.6661682, 436.5891418, -146.6661682, 436.5891418, -583.2550659, 583.2550659
3: -154.5595398, 561.9185181, -154.5595398, 561.9185181, -716.4779663, 716.4779663
4: -131.8280182, 512.9211426, -131.8280182, 512.9211426, -644.7491455, 644.7491455

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9720035, upper bound: 554.9764705
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9772056
time: 1.20 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -123.2541962, 392.5810242, -137.6115265, 437.8572693, -561.1114502, 530.1925659
1: -173.5091553, 394.9635315, -193.6290741, 440.6151428, -614.1242676, 588.5925903
2: -146.6661682, 436.5891418, -163.5594940, 487.1738281, -633.8398438, 600.1486206
3: -154.5595398, 561.9185181, -172.4905701, 626.0422974, -780.6018066, 734.4088745
4: -131.8280182, 512.9211426, -146.9165649, 572.0592651, -703.8872681, 659.8377075

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9720035, upper bound: 554.9782630
time: 1.12 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9789981
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -137.6115265, 437.8572693, -123.2541962, 392.5810242, -530.1925659, 561.1114502
1: -193.6290741, 440.6151428, -173.5091553, 394.9635315, -588.5925293, 614.1242676
2: -163.5594940, 487.1738281, -146.6661682, 436.5891418, -600.1486206, 633.8397827
3: -172.4905701, 626.0422974, -154.5595398, 561.9185181, -734.4088745, 780.6018066
4: -146.9165649, 572.0592651, -131.8280182, 512.9211426, -659.8377075, 703.8872681

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9584742, upper bound: 554.9717693
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9789981, upper bound: 554.9772056
time: 1.27 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -137.6115265, 437.8572693, -137.6115265, 437.8572693, -575.4688110, 575.4688110
1: -193.6290741, 440.6151428, -193.6290741, 440.6151428, -634.2441406, 634.2441406
2: -163.5594940, 487.1738281, -163.5594940, 487.1738281, -650.7333374, 650.7333374
3: -172.4905701, 626.0422974, -172.4905701, 626.0422974, -798.5327759, 798.5327759
4: -146.9165649, 572.0592651, -146.9165649, 572.0592651, -718.9758301, 718.9758301

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9584742, upper bound: 554.9717693
time: 1.38 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9789981, upper bound: 554.9785566
time: 1.21 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.96 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9861204, upper bound: 554.9861084
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9874099, upper bound: 554.9893278
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9861204, upper bound: 554.9861084
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9874099, upper bound: 554.9893278
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9863208, upper bound: 554.9862737
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9873959, upper bound: 554.9873959
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9863208, upper bound: 554.9862737
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9873959, upper bound: 554.9873959
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9842999
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9855172, upper bound: 554.9828575
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9846656
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9855172, upper bound: 554.9833256
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9831599
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9855172, upper bound: 554.9827202
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9856871, upper bound: 554.9831924
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9855172, upper bound: 554.9827752
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9782383, upper bound: 554.9825392
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9570784, upper bound: 554.9776358
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9782383, upper bound: 554.9825392
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9570784, upper bound: 554.9776358
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9628015, upper bound: 554.9797231
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9834286, upper bound: 554.9853005
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9628015, upper bound: 554.9797231
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9834286, upper bound: 554.9853005
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9720035, upper bound: 554.9764705
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9772056
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9720035, upper bound: 554.9782630
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9789981
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9584742, upper bound: 554.9717693
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9789981, upper bound: 554.9772056
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9584742, upper bound: 554.9717693
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -554.9789981, upper bound: 554.9785566

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -107.9422073, 340.9052429, -115.3944626, 363.5161438, -471.4583435, 456.2996826
1: -151.1541901, 344.0680847, -161.7974091, 366.7948303, -517.9489746, 505.8654480
2: -127.8792114, 380.6850891, -136.8181305, 405.7499084, -533.6291504, 517.5031738
3: -134.8178406, 489.5430298, -144.2350464, 521.4130859, -656.2309570, 633.7780762
4: -115.1885223, 447.4972839, -123.1167984, 476.9207153, -592.1092529, 570.6140747

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9848329, upper bound: 554.9848329
time: 1.43 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9848329, upper bound: 554.9861224
time: 1.28 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -115.9024429, 365.3919067, -115.2466278, 363.0174866, -478.9199219, 480.6385193
1: -162.3976135, 368.5887756, -161.5857849, 366.3002625, -528.6978760, 530.1744995
2: -137.2470093, 407.8344116, -136.6386108, 405.2033081, -542.4503174, 544.4729004
3: -144.7448730, 524.3020020, -144.0469360, 520.7048950, -665.4497070, 668.3489380
4: -123.4825363, 479.4687805, -122.9562073, 476.2781677, -599.7606812, 602.4249878

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9861224, upper bound: 554.9880241
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9861224, upper bound: 554.9895137
time: 1.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -107.9422073, 340.9052429, -131.5360107, 414.6815186, -522.6237183, 472.4412537
1: -151.1541901, 344.0680847, -184.4300995, 418.1284180, -569.2824097, 528.4981079
2: -127.8792114, 380.6850891, -155.8180084, 462.7027588, -590.5819702, 536.5029907
3: -134.8178406, 489.5430298, -164.3947754, 593.6577148, -728.4755859, 653.9378052
4: -115.1885223, 447.4972839, -140.0432434, 543.4032593, -658.5917969, 587.5405273

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9849983, upper bound: 554.9850334
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9849983, upper bound: 554.9861084
time: 1.42 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -115.9024429, 365.3919067, -131.3903809, 414.2012939, -530.1037598, 496.7822571
1: -162.3976135, 368.5887756, -184.2218781, 417.6492615, -580.0468140, 552.8106079
2: -137.2470093, 407.8344116, -155.6417542, 462.1729126, -599.4198608, 563.4760742
3: -144.7448730, 524.3020020, -164.2095642, 592.9724121, -737.7171631, 688.5115967
4: -123.4825363, 479.4687805, -139.8851013, 542.7791748, -666.2617188, 619.3538818

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9862878, upper bound: 554.9881620
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9862878, upper bound: 554.9893279
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -124.2585373, 392.6150818, -115.3944626, 363.5161438, -487.7746887, 508.0095215
1: -174.0632782, 396.0002136, -161.7974091, 366.7948303, -540.8580933, 557.7976074
2: -147.1053467, 438.2668457, -136.8181305, 405.7499084, -552.8552246, 575.0849609
3: -155.2312927, 562.5994873, -144.2350464, 521.4130859, -676.6442871, 706.8345337
4: -132.3300018, 514.7772217, -123.1167984, 476.9207153, -609.2507324, 637.8940430

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9850334, upper bound: 554.9849983
time: 1.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9848329, upper bound: 554.9862878
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -131.6705780, 415.2983704, -115.2466278, 363.0174866, -494.6880188, 530.5449829
1: -184.4254608, 418.6448975, -161.5857849, 366.3002625, -550.7257080, 580.2305298
2: -155.7582092, 463.3576355, -136.6386108, 405.2033081, -560.9615479, 599.9961548
3: -164.3948059, 594.8253784, -144.0469360, 520.7048950, -685.0997314, 738.8723145
4: -139.9589539, 544.3573608, -122.9562073, 476.2781677, -616.2371216, 667.3135986

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9861084, upper bound: 554.9861204
time: 1.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9861084, upper bound: 554.9874099
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -124.2585373, 392.6150818, -131.5360107, 414.6815186, -538.9400024, 524.1511230
1: -174.0632782, 396.0002136, -184.4300995, 418.1284180, -592.1917114, 580.4302979
2: -147.1053467, 438.2668457, -155.8180084, 462.7027588, -609.8081055, 594.0848389
3: -155.2312927, 562.5994873, -164.3947754, 593.6577148, -748.8888550, 726.9942627
4: -132.3300018, 514.7772217, -140.0432434, 543.4032593, -675.7332764, 654.8204346

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9851985, upper bound: 554.9851987
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9851985, upper bound: 554.9862737
time: 1.16 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -131.6705780, 415.2983704, -131.3903809, 414.2012939, -545.8717651, 546.6887207
1: -184.4254608, 418.6448975, -184.2218781, 417.6492615, -602.0745850, 602.8666992
2: -155.7582092, 463.3576355, -155.6417542, 462.1729126, -617.9310913, 618.9993286
3: -164.3948059, 594.8253784, -164.2095642, 592.9724121, -757.3671875, 759.0349121
4: -139.9589539, 544.3573608, -139.8851013, 542.7791748, -682.7381592, 684.2424316

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9861084, upper bound: 554.9861204
time: 1.34 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9861084, upper bound: 554.9873959
time: 1.29 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -107.2945023, 338.7147522, -123.1300278, 392.1700439, -499.4645386, 461.8447876
1: -150.3399353, 341.8565674, -173.3341675, 394.5569458, -544.8968506, 515.1907349
2: -127.1170197, 378.1401367, -146.5189209, 436.1359558, -563.2529907, 524.6589966
3: -134.0518646, 486.1622314, -154.4043579, 561.3311157, -695.3828735, 640.5665894
4: -114.4648819, 444.4047546, -131.6971283, 512.3843384, -626.8492432, 576.1018677

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9825532, upper bound: 554.9800191
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9778786, upper bound: 554.9597024
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -113.1604538, 355.7814636, -123.2541962, 392.5810242, -505.7414856, 479.0356445
1: -158.5563812, 359.2316284, -173.5091553, 394.9635315, -553.5198975, 532.7407227
2: -134.0978394, 397.3772888, -146.6661682, 436.5891418, -570.6869507, 544.0432739
3: -141.3645935, 510.6240845, -154.5595398, 561.9185181, -703.2830811, 665.1835938
4: -120.7083435, 467.1063843, -131.8280182, 512.9211426, -633.6294556, 598.9343872

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9820090, upper bound: 554.9779082
time: 1.28 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9778786, upper bound: 554.9573205
time: 1.24 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -107.2945023, 338.7147522, -137.4739380, 437.3987122, -544.6932373, 476.1886902
1: -150.3399353, 341.8565674, -193.4348450, 440.1609192, -590.5008545, 535.2913818
2: -127.1170197, 378.1401367, -163.3966522, 486.6688843, -613.7858887, 541.5368042
3: -134.0518646, 486.1622314, -172.3182831, 625.3876343, -759.4395142, 658.4804688
4: -114.4648819, 444.4047546, -146.7714081, 571.4620972, -685.9270020, 591.1760864

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9825532, upper bound: 554.9801707
time: 1.50 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9797231, upper bound: 554.9637010
time: 1.40 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9853005, upper bound: 554.9841640
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -113.1604538, 355.7814636, -137.6115265, 437.8572693, -551.0177002, 493.3929749
1: -158.5563812, 359.2316284, -193.6290741, 440.6151428, -599.1715088, 552.8605347
2: -134.0978394, 397.3772888, -163.5594940, 487.1738281, -621.2716675, 560.9367676
3: -141.3645935, 510.6240845, -172.4905701, 626.0422974, -767.4068604, 683.1145630
4: -120.7083435, 467.1063843, -146.9165649, 572.0592651, -692.7675781, 614.0229492

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9820090, upper bound: 554.9783096
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9796176, upper bound: 554.9623046
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9850539, upper bound: 554.9828914
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -123.8264008, 391.1094055, -123.1300278, 392.1700439, -515.9964600, 514.2393799
1: -173.4983215, 394.4565735, -173.3341675, 394.5569458, -568.0552979, 567.7907715
2: -146.6014099, 436.4610291, -146.5189209, 436.1359558, -582.7373657, 582.9797974
3: -154.7384491, 560.1813354, -154.4043579, 561.3311157, -716.0695801, 714.5856934
4: -131.8179932, 512.5811157, -131.6971283, 512.3843384, -644.2023315, 644.2782593

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9825392, upper bound: 554.9782383
time: 1.25 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9776254, upper bound: 554.9570784
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -129.3510437, 407.1157227, -123.2541962, 392.5810242, -521.9320679, 530.3699341
1: -181.2432861, 410.7232971, -173.5091553, 394.9635315, -576.2067261, 584.2324219
2: -153.1453857, 454.5035400, -146.6661682, 436.5891418, -589.7344971, 601.1695557
3: -161.5780792, 583.0626831, -154.5595398, 561.9185181, -723.4963989, 737.6221924
4: -137.6862946, 533.7856445, -131.8280182, 512.9211426, -650.6073608, 665.6136475

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9820090, upper bound: 554.9777125
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9776358, upper bound: 554.9566794
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -123.8264008, 391.1094055, -137.4739380, 437.3987122, -561.2250977, 528.5832520
1: -173.4983215, 394.4565735, -193.4348450, 440.1609192, -613.6592407, 587.8914185
2: -146.6014099, 436.4610291, -163.3966522, 486.6688843, -633.2702637, 599.8576050
3: -154.7384491, 560.1813354, -172.3182831, 625.3876343, -780.1260986, 732.4995117
4: -131.8179932, 512.5811157, -146.7714081, 571.4620972, -703.2800903, 659.3525391

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9825392, upper bound: 554.9782383
time: 1.31 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9718957, upper bound: 554.9732059
time: 1.35 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9709695, upper bound: 554.9747038
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -129.3510437, 407.1157227, -137.6115265, 437.8572693, -567.2083130, 544.7272339
1: -181.2432861, 410.7232971, -193.6290741, 440.6151428, -621.8583374, 604.3523560
2: -153.1453857, 454.5035400, -163.5594940, 487.1738281, -640.3192139, 618.0630493
3: -161.5780792, 583.0626831, -172.4905701, 626.0422974, -787.6202393, 755.5531616
4: -137.6862946, 533.7856445, -146.9165649, 572.0592651, -709.7455444, 680.7021484

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9820090, upper bound: 554.9777125
time: 1.31 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9779543, upper bound: 554.9757639
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9770281, upper bound: 554.9754485
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -116.3693695, 371.5358887, -115.3944626, 363.5161438, -479.8854980, 486.9303284
1: -163.6532745, 373.7986450, -161.7974091, 366.7948303, -530.4481201, 535.5959473
2: -138.4061279, 413.3215332, -136.8181305, 405.7499084, -544.1560059, 550.1396484
3: -145.8605499, 532.1776733, -144.2350464, 521.4130859, -667.2735596, 676.4126587
4: -124.5224457, 485.6073914, -123.1167984, 476.9207153, -601.4431152, 608.7241821

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9565663, upper bound: 554.9776308
time: 1.21 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9565663, upper bound: 554.9778786
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -121.5298615, 387.2480164, -115.2466278, 363.0174866, -484.5472717, 502.4946289
1: -171.0144043, 389.6676025, -161.5857849, 366.3002625, -537.3146973, 551.2534180
2: -144.4824677, 430.9747925, -136.6386108, 405.2033081, -549.6856079, 567.6133423
3: -152.3258514, 554.6708984, -144.0469360, 520.7048950, -673.0307617, 698.7178345
4: -129.8988190, 506.4743652, -122.9562073, 476.2781677, -606.1770020, 629.4305420

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9565663, upper bound: 554.9776308
time: 1.18 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9565663, upper bound: 554.9778786
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -116.3693695, 371.5358887, -131.5360107, 414.6815186, -531.0509033, 503.0718994
1: -163.6532745, 373.7986450, -184.4300995, 418.1284180, -581.7816162, 558.2286377
2: -138.4061279, 413.3215332, -155.8180084, 462.7027588, -601.1088867, 569.1395264
3: -145.8605499, 532.1776733, -164.3947754, 593.6577148, -739.5181274, 696.5723877
4: -124.5224457, 485.6073914, -140.0432434, 543.4032593, -667.9257202, 625.6506348

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9565663, upper bound: 554.9776307
time: 1.09 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9565663, upper bound: 554.9776358
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -121.5298615, 387.2480164, -131.3903809, 414.2012939, -535.7310181, 518.6384277
1: -171.0144043, 389.6676025, -184.2218781, 417.6492615, -588.6636963, 573.8894653
2: -144.4824677, 430.9747925, -155.6417542, 462.1729126, -606.6551514, 586.6165161
3: -152.3258514, 554.6708984, -164.2095642, 592.9724121, -745.2982788, 718.8804932
4: -129.8988190, 506.4743652, -139.8851013, 542.7791748, -672.6779785, 646.3594360

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9565663, upper bound: 554.9776307
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9565663, upper bound: 554.9776358
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -127.4574890, 404.6058350, -114.9925232, 362.1365356, -489.5939941, 519.5983887
1: -179.1875305, 407.3471985, -161.2242432, 365.4206848, -544.6082153, 568.5712891
2: -151.3261414, 450.4775696, -136.3343658, 404.2303162, -555.5564575, 586.8119507
3: -159.6564789, 578.5159302, -143.7253876, 519.4346313, -679.0910034, 722.2413330
4: -135.9640045, 528.9044800, -122.6825104, 475.1350403, -611.0990601, 651.5868530

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9063670, upper bound: 554.9514454
time: 1.23 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9637010, upper bound: 554.9797231
time: 1.36 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9623046, upper bound: 554.9796176
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -134.7304993, 428.3551331, -115.3944626, 363.5161438, -498.2466431, 543.7495728
1: -189.5167542, 431.1124573, -161.7974091, 366.7948303, -556.3115845, 592.9097900
2: -160.0974274, 476.7073669, -136.8181305, 405.7499084, -565.8473511, 613.5255127
3: -168.8433228, 612.4936523, -144.2350464, 521.4130859, -690.2564087, 756.7286377
4: -143.8236237, 559.7412109, -123.1167984, 476.9207153, -620.7443237, 682.8579712

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9797896, upper bound: 554.9774197
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9841640, upper bound: 554.9853005
time: 1.32 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9828914, upper bound: 554.9850539
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -127.4574890, 404.6058350, -131.1481781, 413.3646851, -540.8221436, 535.7540283
1: -179.1875305, 407.3471985, -183.8782806, 416.8130798, -596.0006104, 591.2252808
2: -151.3261414, 450.4775696, -155.3522797, 461.2499084, -612.5759888, 605.8298340
3: -159.6564789, 578.5159302, -163.9037476, 591.7586670, -751.4149780, 742.4196777
4: -135.9640045, 528.9044800, -139.6247253, 541.6945801, -677.6585693, 668.5291748

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9056662, upper bound: 554.9488387
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9628015, upper bound: 554.9797231
time: 1.30 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9622487, upper bound: 554.9796176
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -134.7304993, 428.3551331, -131.5360107, 414.6815186, -549.4119873, 559.8911133
1: -189.5167542, 431.1124573, -184.4300995, 418.1284180, -607.6450806, 615.5425415
2: -160.0974274, 476.7073669, -155.8180084, 462.7027588, -622.8001709, 632.5253906
3: -168.8433228, 612.4936523, -164.3947754, 593.6577148, -762.5010376, 776.8883667
4: -143.8236237, 559.7412109, -140.0432434, 543.4032593, -687.2268677, 699.7844238

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9794112, upper bound: 554.9772118
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9772267, upper bound: 554.9758483
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -112.6699219, 357.8710327, -122.8464890, 391.1863098, -503.8561401, 480.7174988
1: -158.3967896, 360.2513428, -172.9276276, 393.5704651, -551.9672241, 533.1789551
2: -133.8872986, 398.2915649, -146.1740112, 435.0499268, -568.9372559, 544.4655151
3: -141.1424103, 512.4251099, -154.0422821, 559.9132080, -701.0556030, 666.4672852
4: -120.4160004, 467.9136658, -131.3858337, 511.1121521, -631.5280762, 599.2994385

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9509640, upper bound: 554.9679649
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9712683, upper bound: 554.9712683
time: 1.26 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9712683, upper bound: 554.9764705
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -120.6354523, 383.9929199, -123.2541962, 392.5810242, -513.2164917, 507.2471313
1: -169.7543335, 386.3460388, -173.5091553, 394.9635315, -564.7178955, 559.8552246
2: -143.5066833, 427.1214600, -146.6661682, 436.5891418, -580.0957031, 573.7874146
3: -151.2345886, 549.6590576, -154.5595398, 561.9185181, -713.1530151, 704.2185669
4: -129.0054932, 501.7817993, -131.8280182, 512.9211426, -641.9265747, 633.6098022

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9764705, upper bound: 554.9720035
time: 1.25 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9764705, upper bound: 554.9772056
time: 1.31 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -112.6699219, 357.8710327, -137.2233429, 436.5267639, -549.1966553, 495.0943298
1: -158.3967896, 360.2513428, -193.0767212, 439.2861938, -597.6828613, 553.3280640
2: -133.8872986, 398.2915649, -163.0932922, 485.7078552, -619.5951538, 561.3848877
3: -141.1424103, 512.4251099, -172.0001221, 624.1306763, -765.2730713, 684.4252319
4: -120.4160004, 467.9136658, -146.4967651, 570.3374023, -690.7532959, 614.4104004

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9509640, upper bound: 554.9696513
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9530379, upper bound: 554.9530379
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9530379, upper bound: 554.9782630
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -120.6354523, 383.9929199, -137.6115265, 437.8572693, -558.4927368, 521.6044312
1: -169.7543335, 386.3460388, -193.6290741, 440.6151428, -610.3695068, 579.9750366
2: -143.5066833, 427.1214600, -163.5594940, 487.1738281, -630.6804810, 590.6809692
3: -151.2345886, 549.6590576, -172.4905701, 626.0422974, -777.2768555, 722.1495361
4: -129.0054932, 501.7817993, -146.9165649, 572.0592651, -701.0647583, 648.6983643

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9530378, upper bound: 554.9530379
time: 1.20 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9530379, upper bound: 554.9789981
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -127.4574890, 404.6058350, -122.8464890, 391.1863098, -518.6437988, 527.4523315
1: -179.1875305, 407.3471985, -172.9276276, 393.5704651, -572.7578735, 580.2747192
2: -151.3261414, 450.4775696, -146.1740112, 435.0499268, -586.3760376, 596.6514893
3: -159.6564789, 578.5159302, -154.0422821, 559.9132080, -719.5695801, 732.5581055
4: -135.9640045, 528.9044800, -131.3858337, 511.1121521, -647.0760498, 660.2902832

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9043255, upper bound: 554.9493992
time: 1.58 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9577391, upper bound: 554.9665672
time: 1.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9577391, upper bound: 554.9717693
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -134.7304993, 428.3551331, -123.2541962, 392.5810242, -527.3115234, 551.6093140
1: -189.5167542, 431.1124573, -173.5091553, 394.9635315, -584.4802856, 604.6215820
2: -160.0974274, 476.7073669, -146.6661682, 436.5891418, -596.6865234, 623.3733521
3: -168.8433228, 612.4936523, -154.5595398, 561.9185181, -730.7618408, 767.0531006
4: -143.8236237, 559.7412109, -131.8280182, 512.9211426, -656.7447510, 691.5692139

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9749807, upper bound: 554.9708778
time: 1.48 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9782630, upper bound: 554.9720035
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9782630, upper bound: 554.9772056
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -127.4574890, 404.6058350, -137.2233429, 436.5267639, -563.9842529, 541.8291626
1: -179.1875305, 407.3471985, -193.0767212, 439.2861938, -618.4736938, 600.4238281
2: -151.3261414, 450.4775696, -163.0932922, 485.7078552, -637.0339355, 613.5708618
3: -159.6564789, 578.5159302, -172.0001221, 624.1306763, -783.7870483, 750.5160522
4: -135.9640045, 528.9044800, -146.4967651, 570.3374023, -706.3012695, 675.4012451

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9043106, upper bound: 554.9488592
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9000538, upper bound: 554.9418211
time: 1.19 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -134.7304993, 428.3551331, -137.6115265, 437.8572693, -572.5877686, 565.9666748
1: -189.5167542, 431.1124573, -193.6290741, 440.6151428, -630.1318359, 624.7414551
2: -160.0974274, 476.7073669, -163.5594940, 487.1738281, -647.2712402, 640.2668457
3: -168.8433228, 612.4936523, -172.4905701, 626.0422974, -794.8856201, 784.9840698
4: -143.8236237, 559.7412109, -146.9165649, 572.0592651, -715.8828735, 706.6577759

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9749807, upper bound: 554.9710369
time: 1.49 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9742533, upper bound: 554.9727259
time: 0.94 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.49 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9848329, upper bound: 554.9848329
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9848329, upper bound: 554.9861224
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9861224, upper bound: 554.9880241
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9861224, upper bound: 554.9895137
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9849983, upper bound: 554.9850334
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9849983, upper bound: 554.9861084
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9862878, upper bound: 554.9881620
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9862878, upper bound: 554.9893279
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9850334, upper bound: 554.9849983
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9848329, upper bound: 554.9862878
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9861084, upper bound: 554.9861204
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9861084, upper bound: 554.9874099
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9851985, upper bound: 554.9851987
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9851985, upper bound: 554.9862737
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9861084, upper bound: 554.9861204
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9861084, upper bound: 554.9873959
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9825532, upper bound: 554.9800191
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9778786, upper bound: 554.9597024
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9820090, upper bound: 554.9779082
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9778786, upper bound: 554.9573205
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9797231, upper bound: 554.9637010
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9853005, upper bound: 554.9841640
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9796176, upper bound: 554.9623046
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9850539, upper bound: 554.9828914
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9825392, upper bound: 554.9782383
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9776254, upper bound: 554.9570784
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9820090, upper bound: 554.9777125
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9776358, upper bound: 554.9566794
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9718957, upper bound: 554.9732059
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9709695, upper bound: 554.9747038
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9779543, upper bound: 554.9757639
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9770281, upper bound: 554.9754485
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9565663, upper bound: 554.9776308
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9565663, upper bound: 554.9778786
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9565663, upper bound: 554.9776308
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9565663, upper bound: 554.9778786
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9565663, upper bound: 554.9776307
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9565663, upper bound: 554.9776358
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9565663, upper bound: 554.9776307
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9565663, upper bound: 554.9776358
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9637010, upper bound: 554.9797231
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9623046, upper bound: 554.9796176
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9841640, upper bound: 554.9853005
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9828914, upper bound: 554.9850539
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9628015, upper bound: 554.9797231
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9622487, upper bound: 554.9796176
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9794112, upper bound: 554.9772118
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9772267, upper bound: 554.9758483
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9712683, upper bound: 554.9712683
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9712683, upper bound: 554.9764705
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9764705, upper bound: 554.9720035
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9764705, upper bound: 554.9772056
IS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9530379, upper bound: 554.9530379
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9530379, upper bound: 554.9782630
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9530378, upper bound: 554.9530379
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9530379, upper bound: 554.9789981
IS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9577391, upper bound: 554.9665672
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9577391, upper bound: 554.9717693
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9782630, upper bound: 554.9720035
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9782630, upper bound: 554.9772056
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9043106, upper bound: 554.9488592
IS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9000538, upper bound: 554.9418211
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9749807, upper bound: 554.9710369
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9742533, upper bound: 554.9727259

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -107.9422073, 340.9052429, -107.9422073, 340.9052429, -448.8474426, 448.8474426
1: -151.1541901, 344.0680847, -151.1541901, 344.0680847, -495.2222595, 495.2222595
2: -127.8792114, 380.6850891, -127.8792114, 380.6850891, -508.5643005, 508.5643005
3: -134.8178406, 489.5430298, -134.8178406, 489.5430298, -624.3608398, 624.3608398
4: -115.1885223, 447.4972839, -115.1885223, 447.4972839, -562.6857910, 562.6857910

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9841677, upper bound: 554.9845104
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9848329, upper bound: 554.9848329
time: 1.34 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -107.9422073, 340.9052429, -115.9024429, 365.3919067, -473.3341064, 456.8076782
1: -151.1541901, 344.0680847, -162.3976135, 368.5887756, -519.7427979, 506.4656677
2: -127.8792114, 380.6850891, -137.2470093, 407.8344116, -535.7135620, 517.9319458
3: -134.8178406, 489.5430298, -144.7448730, 524.3020020, -659.1198730, 634.2878418
4: -115.1885223, 447.4972839, -123.4825363, 479.4687805, -594.6572876, 570.9797974

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9841677, upper bound: 554.9852556
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9848329, upper bound: 554.9855782
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -115.9024429, 365.3919067, -107.9422073, 340.9052429, -456.8076782, 473.3340759
1: -162.3976135, 368.5887756, -151.1541901, 344.0680847, -506.4656677, 519.7427979
2: -137.2470093, 407.8344116, -127.8792114, 380.6850891, -517.9319458, 535.7135620
3: -144.7448730, 524.3020020, -134.8178406, 489.5430298, -634.2878418, 659.1198730
4: -123.4825363, 479.4687805, -115.1885223, 447.4972839, -570.9797974, 594.6572876

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9861224, upper bound: 554.9880241
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855782, upper bound: 554.9857904
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -115.9024429, 365.3919067, -115.9024429, 365.3919067, -481.2943420, 481.2943420
1: -162.3976135, 368.5887756, -162.3976135, 368.5887756, -530.9862671, 530.9863281
2: -137.2470093, 407.8344116, -137.2470093, 407.8344116, -545.0814209, 545.0814209
3: -144.7448730, 524.3020020, -144.7448730, 524.3020020, -669.0468750, 669.0468750
4: -123.4825363, 479.4687805, -123.4825363, 479.4687805, -602.9512939, 602.9512939

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9861224, upper bound: 554.9886930
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855782, upper bound: 554.9865356
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -107.9422073, 340.9052429, -124.2585373, 392.6150818, -500.5572815, 465.1637878
1: -151.1541901, 344.0680847, -174.0632782, 396.0002136, -547.1543579, 518.1313477
2: -127.8792114, 380.6850891, -147.1053467, 438.2668457, -566.1460571, 527.7903442
3: -134.8178406, 489.5430298, -155.2312927, 562.5994873, -697.4173584, 644.7742920
4: -115.1885223, 447.4972839, -132.3300018, 514.7772217, -629.9657593, 579.8272705

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9843330, upper bound: 554.9847108
time: 1.45 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9849983, upper bound: 554.9850334
time: 1.27 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -107.9422073, 340.9052429, -131.6705780, 415.2983704, -523.2406006, 472.5758057
1: -151.1541901, 344.0680847, -184.4254608, 418.6448975, -569.7988892, 528.4935303
2: -127.8792114, 380.6850891, -155.7582092, 463.3576355, -591.2368164, 536.4431763
3: -134.8178406, 489.5430298, -164.3948059, 594.8253784, -729.6431885, 653.9378662
4: -115.1885223, 447.4972839, -139.9589539, 544.3573608, -659.5458984, 587.4562378

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9843330, upper bound: 554.9852556
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9849983, upper bound: 554.9855782
time: 1.46 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -115.9024429, 365.3919067, -124.2585373, 392.6150818, -508.5175171, 489.6504211
1: -162.3976135, 368.5887756, -174.0632782, 396.0002136, -558.3978271, 542.6519775
2: -137.2470093, 407.8344116, -147.1053467, 438.2668457, -575.5138550, 554.9397583
3: -144.7448730, 524.3020020, -155.2312927, 562.5994873, -707.3442993, 679.5332642
4: -123.4825363, 479.4687805, -132.3300018, 514.7772217, -638.2597656, 611.7987671

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9862878, upper bound: 554.9881620
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9857435, upper bound: 554.9859908
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -115.9024429, 365.3919067, -131.6705780, 415.2983704, -531.2008057, 497.0624390
1: -162.3976135, 368.5887756, -184.4254608, 418.6448975, -581.0424194, 553.0140381
2: -137.2470093, 407.8344116, -155.7582092, 463.3576355, -600.6046143, 563.5926514
3: -144.7448730, 524.3020020, -164.3948059, 594.8253784, -739.5701294, 688.6967773
4: -123.4825363, 479.4687805, -139.9589539, 544.3573608, -667.8399048, 619.4277344

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9862878, upper bound: 554.9886930
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9857435, upper bound: 554.9865356
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -124.2585373, 392.6150818, -107.9422073, 340.9052429, -465.1637878, 500.5572815
1: -174.0632782, 396.0002136, -151.1541901, 344.0680847, -518.1313477, 547.1542969
2: -147.1053467, 438.2668457, -127.8792114, 380.6850891, -527.7903442, 566.1460571
3: -155.2312927, 562.5994873, -134.8178406, 489.5430298, -644.7742920, 697.4173584
4: -132.3300018, 514.7772217, -115.1885223, 447.4972839, -579.8272705, 629.9657593

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9850334, upper bound: 554.9849564
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9850334, upper bound: 554.9849983
time: 1.27 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -124.2585373, 392.6150818, -115.9024429, 365.3919067, -489.6504211, 508.5175171
1: -174.0632782, 396.0002136, -162.3976135, 368.5887756, -542.6519775, 558.3978271
2: -147.1053467, 438.2668457, -137.2470093, 407.8344116, -554.9397583, 575.5138550
3: -155.2312927, 562.5994873, -144.7448730, 524.3020020, -679.5332642, 707.3442993
4: -132.3300018, 514.7772217, -123.4825363, 479.4687805, -611.7987671, 638.2597656

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9850334, upper bound: 554.9857017
time: 1.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9850334, upper bound: 554.9857435
time: 1.25 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -131.6705780, 415.2983704, -107.9422073, 340.9052429, -472.5758057, 523.2406006
1: -184.4254608, 418.6448975, -151.1541901, 344.0680847, -528.4935303, 569.7988892
2: -155.7582092, 463.3576355, -127.8792114, 380.6850891, -536.4431763, 591.2368164
3: -164.3948059, 594.8253784, -134.8178406, 489.5430298, -653.9378662, 729.6431885
4: -139.9589539, 544.3573608, -115.1885223, 447.4972839, -587.4562378, 659.5458984

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9861084, upper bound: 554.9861204
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855782, upper bound: 554.9855946
time: 1.31 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -131.6705780, 415.2983704, -115.9024429, 365.3919067, -497.0624390, 531.2008057
1: -184.4254608, 418.6448975, -162.3976135, 368.5887756, -553.0140991, 581.0424194
2: -155.7582092, 463.3576355, -137.2470093, 407.8344116, -563.5926514, 600.6046143
3: -164.3948059, 594.8253784, -144.7448730, 524.3020020, -688.6967773, 739.5701294
4: -139.9589539, 544.3573608, -123.4825363, 479.4687805, -619.4277344, 667.8399048

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9861084, upper bound: 554.9868656
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855782, upper bound: 554.9863399
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -124.2585373, 392.6150818, -124.2585373, 392.6150818, -516.8735962, 516.8735962
1: -174.0632782, 396.0002136, -174.0632782, 396.0002136, -570.0634766, 570.0634766
2: -147.1053467, 438.2668457, -147.1053467, 438.2668457, -585.3721924, 585.3721924
3: -155.2312927, 562.5994873, -155.2312927, 562.5994873, -717.8307495, 717.8307495
4: -132.3300018, 514.7772217, -132.3300018, 514.7772217, -647.1072388, 647.1072388

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9851985, upper bound: 554.9851568
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9851985, upper bound: 554.9851987
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -124.2585373, 392.6150818, -131.6705780, 415.2983704, -539.5567627, 524.2856445
1: -174.0632782, 396.0002136, -184.4254608, 418.6448975, -592.7081299, 580.4256592
2: -147.1053467, 438.2668457, -155.7582092, 463.3576355, -610.4630127, 594.0250244
3: -155.2312927, 562.5994873, -164.3948059, 594.8253784, -750.0565796, 726.9942627
4: -132.3300018, 514.7772217, -139.9589539, 544.3573608, -676.6873779, 654.7362061

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9851985, upper bound: 554.9857017
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9851985, upper bound: 554.9857435
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -131.6705780, 415.2983704, -124.2585373, 392.6150818, -524.2856445, 539.5567627
1: -184.4254608, 418.6448975, -174.0632782, 396.0002136, -580.4255981, 592.7081299
2: -155.7582092, 463.3576355, -147.1053467, 438.2668457, -594.0250244, 610.4630127
3: -164.3948059, 594.8253784, -155.2312927, 562.5994873, -726.9942627, 750.0565796
4: -139.9589539, 544.3573608, -132.3300018, 514.7772217, -654.7362061, 676.6873779

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9861084, upper bound: 554.9861204
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855782, upper bound: 554.9855946
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -131.6705780, 415.2983704, -131.6705780, 415.2983704, -546.9688721, 546.9688721
1: -184.4254608, 418.6448975, -184.4254608, 418.6448975, -603.0701294, 603.0701294
2: -155.7582092, 463.3576355, -155.7582092, 463.3576355, -619.1158447, 619.1158447
3: -164.3948059, 594.8253784, -164.3948059, 594.8253784, -759.2202148, 759.2202148
4: -139.9589539, 544.3573608, -139.9589539, 544.3573608, -684.3162842, 684.3162842

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9861084, upper bound: 554.9868656
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855782, upper bound: 554.9863399
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -107.2945023, 338.7147522, -116.2401276, 371.1069336, -478.4014282, 454.9548645
1: -150.3399353, 341.8565674, -163.4711304, 373.3738403, -523.7137451, 505.3276062
2: -127.1170197, 378.1401367, -138.2529144, 412.8448486, -539.9618530, 516.3930054
3: -134.0518646, 486.1622314, -145.6987762, 531.5642700, -665.6160278, 631.8610229
4: -114.4648819, 444.4047546, -124.3859863, 485.0414734, -599.5063477, 568.7907104

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9770369, upper bound: 554.9564025
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9770369, upper bound: 554.9597024
time: 1.20 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.52 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9841677, upper bound: 554.9845104
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9848329, upper bound: 554.9848329
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9841677, upper bound: 554.9852556
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9848329, upper bound: 554.9855782
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9861224, upper bound: 554.9880241
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9855782, upper bound: 554.9857904
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9861224, upper bound: 554.9886930
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9855782, upper bound: 554.9865356
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9843330, upper bound: 554.9847108
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9849983, upper bound: 554.9850334
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9843330, upper bound: 554.9852556
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9849983, upper bound: 554.9855782
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9862878, upper bound: 554.9881620
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9857435, upper bound: 554.9859908
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9862878, upper bound: 554.9886930
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9857435, upper bound: 554.9865356
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9850334, upper bound: 554.9849564
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9850334, upper bound: 554.9849983
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9850334, upper bound: 554.9857017
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9850334, upper bound: 554.9857435
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9861084, upper bound: 554.9861204
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9855782, upper bound: 554.9855946
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9861084, upper bound: 554.9868656
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9855782, upper bound: 554.9863399
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9851985, upper bound: 554.9851568
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9851985, upper bound: 554.9851987
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9851985, upper bound: 554.9857017
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9851985, upper bound: 554.9857435
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9861084, upper bound: 554.9861204
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9855782, upper bound: 554.9855946
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9861084, upper bound: 554.9868656
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9855782, upper bound: 554.9863399
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9770369, upper bound: 554.9564025
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -554.9770369, upper bound: 554.9597024
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9778786, upper bound: 554.9597024
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9820090, upper bound: 554.9779082
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9778786, upper bound: 554.9573205
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9797231, upper bound: 554.9637010
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9853005, upper bound: 554.9841640
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9796176, upper bound: 554.9623046
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9850539, upper bound: 554.9828914
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9825392, upper bound: 554.9782383
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9776254, upper bound: 554.9570784
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9820090, upper bound: 554.9777125
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9776358, upper bound: 554.9566794
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9718957, upper bound: 554.9732059
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9709695, upper bound: 554.9747038
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9779543, upper bound: 554.9757639
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9770281, upper bound: 554.9754485
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9565663, upper bound: 554.9776308
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9565663, upper bound: 554.9778786
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9565663, upper bound: 554.9776308
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9565663, upper bound: 554.9778786
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9565663, upper bound: 554.9776307
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9565663, upper bound: 554.9776358
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9565663, upper bound: 554.9776307
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9565663, upper bound: 554.9776358
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9637010, upper bound: 554.9797231
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9623046, upper bound: 554.9796176
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9841640, upper bound: 554.9853005
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9828914, upper bound: 554.9850539
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9628015, upper bound: 554.9797231
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9622487, upper bound: 554.9796176
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9794112, upper bound: 554.9772118
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9772267, upper bound: 554.9758483
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9712683, upper bound: 554.9712683
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9712683, upper bound: 554.9764705
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9764705, upper bound: 554.9720035
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9764705, upper bound: 554.9772056
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9530379, upper bound: 554.9782630
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9530379, upper bound: 554.9789981
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9577391, upper bound: 554.9717693
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9782630, upper bound: 554.9720035
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9782630, upper bound: 554.9772056
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9749807, upper bound: 554.9710369
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 0, lower bound: -554.9742533, upper bound: 554.9727259
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=619.6494140625
rel_dist={0: [-554.9911089992902, 554.9911089992902]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9839967
time: 1.32 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9794499, upper bound: 554.9794499
time: 1.01 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.50 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.50
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9839967
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.50
Output dim: 0, lower bound: -554.9794499, upper bound: 554.9794499

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -135.4273682, 427.6317749, -146.9334869, 467.3382263, -602.7656250, 574.5652466
1: -189.9990234, 430.8988953, -206.6879730, 470.1475525, -660.1466064, 637.5866699
2: -160.4833527, 476.7970886, -174.5671082, 519.8568115, -680.3401489, 651.3641968
3: -169.3506470, 612.0695801, -184.1661987, 668.0075684, -837.3581543, 796.2357178
4: -144.2209015, 559.9951782, -156.8076019, 610.4512329, -754.6721191, 716.8027954

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9828438
time: 0.90 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9833717
time: 1.07 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -141.9587555, 452.3752747, -146.2898865, 465.6584473, -607.6171875, 598.6651611
1: -199.8379517, 454.9771729, -205.8313904, 468.4003906, -668.2383423, 660.8085938
2: -168.7648926, 503.0252075, -173.8343811, 517.9138184, -686.6785889, 676.8594971
3: -178.0149994, 646.6646729, -183.3922729, 665.6176147, -843.6326294, 830.0568848
4: -151.5778503, 590.7100830, -156.1433868, 608.1997070, -759.7774658, 746.8533936

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9794499, upper bound: 554.9789849
time: 1.34 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9794499, upper bound: 554.9794499
time: 0.94 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.30 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.30
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9828438
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.30
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9833717
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.30
Output dim: 0, lower bound: -554.9794499, upper bound: 554.9789849
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.30
Output dim: 0, lower bound: -554.9794499, upper bound: 554.9794499

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -133.5709229, 421.4596252, -127.8818512, 406.5073853, -540.0783081, 549.3414917
1: -187.3936005, 424.7517395, -179.9052582, 409.1180725, -596.5116577, 604.6569214
2: -158.3083801, 469.9678345, -152.0791168, 452.2377625, -610.5461426, 622.0468750
3: -167.0257874, 603.2378540, -160.3091431, 581.8273315, -748.8530884, 763.5468750
4: -142.2788239, 551.9683838, -136.7295532, 531.3318481, -673.6106567, 688.6979370

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9828438
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9828438
time: 1.08 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -134.1717834, 423.4424744, -142.5935822, 452.7910767, -586.9628906, 566.0359497
1: -188.2073669, 426.7677002, -200.4949493, 455.7381287, -643.9454956, 627.2625122
2: -158.9820557, 472.2454224, -169.3762360, 503.9754333, -662.9575195, 641.6216431
3: -167.7543640, 606.1092529, -178.6541748, 647.3225098, -815.0768433, 784.7634277
4: -142.8764801, 554.6398315, -152.1581573, 591.7705688, -734.6470337, 706.7977905

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9833717
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9833717
time: 1.30 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -140.4029388, 447.2006531, -127.3104095, 405.0379333, -545.4408569, 574.5110474
1: -197.6610413, 449.7850647, -179.1357727, 407.5769348, -605.2379150, 628.9208374
2: -166.9412231, 497.2776794, -151.4236908, 450.5628052, -617.5039673, 648.7013550
3: -176.0716705, 639.2141724, -159.6134338, 579.7385864, -755.8102417, 798.8276367
4: -149.9378967, 583.9582520, -136.1260223, 529.3653564, -679.3032227, 720.0842896

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9789849, upper bound: 554.9789849
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9789849, upper bound: 554.9789849
time: 1.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -140.5934448, 447.8157349, -142.0003204, 451.2975464, -591.8909302, 589.8160400
1: -197.8878479, 450.4583435, -199.7085114, 454.1860352, -652.0738525, 650.1668701
2: -167.1294403, 498.0463867, -168.7018433, 502.2332153, -669.3626709, 666.7482300
3: -176.2795715, 640.1743774, -177.9436188, 645.2158203, -821.4953613, 818.1179810
4: -150.1146088, 584.8482666, -151.5470428, 589.7641602, -739.8787842, 736.3952637

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9789849, upper bound: 554.9794499
time: 1.19 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9789849, upper bound: 554.9794499
time: 1.13 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.64 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.64
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9828438
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.64
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9828438
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.64
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9833717
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.64
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9833717
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.64
Output dim: 0, lower bound: -554.9789849, upper bound: 554.9789849
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.64
Output dim: 0, lower bound: -554.9789849, upper bound: 554.9789849
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.64
Output dim: 0, lower bound: -554.9789849, upper bound: 554.9794499
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.64
Output dim: 0, lower bound: -554.9789849, upper bound: 554.9794499

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -115.3944626, 363.5161438, -127.8818512, 406.5073853, -521.9018555, 491.3980103
1: -161.7974091, 366.7948303, -179.9052582, 409.1180725, -570.9153442, 546.7000732
2: -136.8181305, 405.7499084, -152.0791168, 452.2377625, -589.0559082, 557.8290405
3: -144.2350464, 521.4130859, -160.3091431, 581.8273315, -726.0623779, 681.7221069
4: -123.1167984, 476.9207153, -136.7295532, 531.3318481, -654.4486694, 613.6502686

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9828438
time: 1.57 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9828438
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -131.5360107, 414.6815186, -127.8818512, 406.5073853, -538.0433960, 542.5633545
1: -184.4300995, 418.1284180, -179.9052582, 409.1180725, -593.5481567, 598.0335083
2: -155.8180084, 462.7027588, -152.0791168, 452.2377625, -608.0557861, 614.7818604
3: -164.3947754, 593.6577148, -160.3091431, 581.8273315, -746.2221069, 753.9666748
4: -140.0432434, 543.4032593, -136.7295532, 531.3318481, -671.3751221, 680.1328125

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9828438
time: 1.34 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9828438
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -115.3944626, 363.5161438, -142.5935822, 452.7910767, -568.1855469, 506.1097107
1: -161.7974091, 366.7948303, -200.4949493, 455.7381287, -617.5354004, 567.2897949
2: -136.8181305, 405.7499084, -169.3762360, 503.9754333, -640.7935791, 575.1261597
3: -144.2350464, 521.4130859, -178.6541748, 647.3225098, -791.5575562, 700.0672607
4: -123.1167984, 476.9207153, -152.1581573, 591.7705688, -714.8873291, 629.0788574

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9833717
time: 1.22 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9833717
time: 1.36 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -131.5360107, 414.6815186, -142.5935822, 452.7910767, -584.3270874, 557.2750854
1: -184.4300995, 418.1284180, -200.4949493, 455.7381287, -640.1681519, 618.6231689
2: -155.8180084, 462.7027588, -169.3762360, 503.9754333, -659.7934570, 632.0789185
3: -164.3947754, 593.6577148, -178.6541748, 647.3225098, -811.7172852, 772.3118286
4: -140.0432434, 543.4032593, -152.1581573, 591.7705688, -731.8138428, 695.5614014

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9829312
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9829312
time: 1.31 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -123.2541962, 392.5810242, -127.3104095, 405.0379333, -528.2921143, 519.8914185
1: -173.5091553, 394.9635315, -179.1357727, 407.5769348, -581.0860596, 574.0993042
2: -146.6661682, 436.5891418, -151.4236908, 450.5628052, -597.2287598, 588.0128174
3: -154.5595398, 561.9185181, -159.6134338, 579.7385864, -734.2980957, 721.5319214
4: -131.8280182, 512.9211426, -136.1260223, 529.3653564, -661.1933594, 649.0471802

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9789850, upper bound: 554.9789849
time: 1.52 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9789850, upper bound: 554.9789849
time: 1.34 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -137.6115265, 437.8572693, -127.3104095, 405.0379333, -542.6494751, 565.1676636
1: -193.6290741, 440.6151428, -179.1357727, 407.5769348, -601.2058716, 619.7509155
2: -163.5594940, 487.1738281, -151.4236908, 450.5628052, -614.1223145, 638.5975342
3: -172.4905701, 626.0422974, -159.6134338, 579.7385864, -752.2290649, 785.6557617
4: -146.9165649, 572.0592651, -136.1260223, 529.3653564, -676.2819214, 708.1853027

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9789850, upper bound: 554.9789849
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9789850, upper bound: 554.9789849
time: 1.19 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -123.2539520, 392.5803528, -142.0003204, 451.2975464, -574.5513916, 534.5806885
1: -173.5087891, 394.9628601, -199.7085114, 454.1860352, -627.6948242, 594.6713867
2: -146.6658630, 436.5884094, -168.7018433, 502.2332153, -648.8990479, 605.2902222
3: -154.5592194, 561.9174805, -177.9436188, 645.2158203, -799.7750244, 739.8610840
4: -131.8277740, 512.9202881, -151.5470428, 589.7641602, -721.5918579, 664.4672852

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9789849, upper bound: 554.9794499
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9789849, upper bound: 554.9794499
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -137.6115265, 437.8572693, -142.0003204, 451.2975464, -588.9090576, 579.8576050
1: -193.6290741, 440.6151428, -199.7085114, 454.1860352, -647.8150635, 640.3236084
2: -163.5594940, 487.1738281, -168.7018433, 502.2332153, -665.7927246, 655.8756714
3: -172.4905701, 626.0422974, -177.9436188, 645.2158203, -817.7063599, 803.9859009
4: -146.9165649, 572.0592651, -151.5470428, 589.7641602, -736.6806641, 723.6062622

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9789849, upper bound: 554.9790678
time: 1.13 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9789849, upper bound: 554.9790678
time: 1.17 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.43 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9828438
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9828438
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9828438
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9828438
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9833717
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9833717
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9829312
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9829312
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -554.9789850, upper bound: 554.9789849
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -554.9789850, upper bound: 554.9789849
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -554.9789850, upper bound: 554.9789849
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -554.9789850, upper bound: 554.9789849
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -554.9789849, upper bound: 554.9794499
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -554.9789849, upper bound: 554.9794499
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -554.9789849, upper bound: 554.9790678
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -554.9789849, upper bound: 554.9790678

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -115.3944626, 363.5161438, -115.3944626, 363.5161438, -478.9105835, 478.9105835
1: -161.7974091, 366.7948303, -161.7974091, 366.7948303, -528.5922241, 528.5922241
2: -136.8181305, 405.7499084, -136.8181305, 405.7499084, -542.5680542, 542.5680542
3: -144.2350464, 521.4130859, -144.2350464, 521.4130859, -665.6481323, 665.6481323
4: -123.1167984, 476.9207153, -123.1167984, 476.9207153, -600.0375366, 600.0375366

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9686706, upper bound: 554.9541939
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9693960, upper bound: 554.9567523
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -115.3944626, 363.5161438, -123.2541962, 392.5810242, -507.9754639, 486.7703247
1: -161.7974091, 366.7948303, -173.5091553, 394.9635315, -556.7608643, 540.3039551
2: -136.8181305, 405.7499084, -146.6661682, 436.5891418, -573.4072266, 552.4159546
3: -144.2350464, 521.4130859, -154.5595398, 561.9185181, -706.1535034, 675.9725952
4: -123.1167984, 476.9207153, -131.8280182, 512.9211426, -636.0379028, 608.7487183

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9686706, upper bound: 554.9541939
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9693960, upper bound: 554.9567523
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -131.5360107, 414.6815186, -115.3944626, 363.5161438, -495.0521545, 530.0759888
1: -184.4300995, 418.1284180, -161.7974091, 366.7948303, -551.2249146, 579.9256592
2: -155.8180084, 462.7027588, -136.8181305, 405.7499084, -561.5679321, 599.5208740
3: -164.3947754, 593.6577148, -144.2350464, 521.4130859, -685.8078613, 737.8927002
4: -140.0432434, 543.4032593, -123.1167984, 476.9207153, -616.9639893, 666.5200806

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9345542, upper bound: 554.9149896
time: 1.29 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9828438
time: 1.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855075, upper bound: 554.9825508
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -131.5360107, 414.6815186, -123.2541962, 392.5810242, -524.1170654, 537.9357300
1: -184.4300995, 418.1284180, -173.5091553, 394.9635315, -579.3936157, 591.6375732
2: -155.8180084, 462.7027588, -146.6661682, 436.5891418, -592.4070435, 609.3687744
3: -164.3947754, 593.6577148, -154.5595398, 561.9185181, -726.3132324, 748.2171631
4: -140.0432434, 543.4032593, -131.8280182, 512.9211426, -652.9643555, 675.2312622

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9345542, upper bound: 554.9149896
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9828438
time: 1.29 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855075, upper bound: 554.9825508
time: 1.83 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -115.3944626, 363.5161438, -131.5360107, 414.6815186, -530.0759888, 495.0521545
1: -161.7974091, 366.7948303, -184.4300995, 418.1284180, -579.9256592, 551.2249146
2: -136.8181305, 405.7499084, -155.8180084, 462.7027588, -599.5208740, 561.5679321
3: -144.2350464, 521.4130859, -164.3947754, 593.6577148, -737.8927002, 685.8078613
4: -123.1167984, 476.9207153, -140.0432434, 543.4032593, -666.5200806, 616.9639893

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9778722, upper bound: 554.9788406
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9839967
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855023, upper bound: 554.9831879
time: 1.27 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -115.3944626, 363.5161438, -137.6115265, 437.8572693, -553.2517090, 501.1276550
1: -161.7974091, 366.7948303, -193.6290741, 440.6151428, -602.4124146, 560.4238892
2: -136.8181305, 405.7499084, -163.5594940, 487.1738281, -623.9919434, 569.3093872
3: -144.2350464, 521.4130859, -172.4905701, 626.0422974, -770.2773438, 693.9035034
4: -123.1167984, 476.9207153, -146.9165649, 572.0592651, -695.1760864, 623.8372803

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9778722, upper bound: 554.9788406
time: 1.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9839967
time: 1.85 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855023, upper bound: 554.9831879
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -131.5360107, 414.6815186, -131.5360107, 414.6815186, -546.2175293, 546.2175293
1: -184.4300995, 418.1284180, -184.4300995, 418.1284180, -602.5584106, 602.5584717
2: -155.8180084, 462.7027588, -155.8180084, 462.7027588, -618.5207520, 618.5207520
3: -164.3947754, 593.6577148, -164.3947754, 593.6577148, -758.0524292, 758.0524292
4: -140.0432434, 543.4032593, -140.0432434, 543.4032593, -683.4465332, 683.4465332

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9829312
time: 1.26 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855044, upper bound: 554.9826638
time: 1.29 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -131.5360107, 414.6815186, -137.6115265, 437.8572693, -569.3933105, 552.2930298
1: -184.4300995, 418.1284180, -193.6290741, 440.6151428, -625.0452271, 611.7573853
2: -155.8180084, 462.7027588, -163.5594940, 487.1738281, -642.9918213, 626.2622681
3: -164.3947754, 593.6577148, -172.4905701, 626.0422974, -790.4370728, 766.1481323
4: -140.0432434, 543.4032593, -146.9165649, 572.0592651, -712.1025391, 690.3198242

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9829312
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855044, upper bound: 554.9826638
time: 1.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -123.2541962, 392.5810242, -115.3944626, 363.5161438, -486.7703247, 507.9754639
1: -173.5091553, 394.9635315, -161.7974091, 366.7948303, -540.3039551, 556.7608643
2: -146.6661682, 436.5891418, -136.8181305, 405.7499084, -552.4159546, 573.4072266
3: -154.5595398, 561.9185181, -144.2350464, 521.4130859, -675.9725952, 706.1535034
4: -131.8280182, 512.9211426, -123.1167984, 476.9207153, -608.7487183, 636.0379028

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9646884, upper bound: 554.9517015
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9495942, upper bound: 554.9495942
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -123.2541962, 392.5810242, -123.2541962, 392.5810242, -515.8352051, 515.8352051
1: -173.5091553, 394.9635315, -173.5091553, 394.9635315, -568.4726562, 568.4726562
2: -146.6661682, 436.5891418, -146.6661682, 436.5891418, -583.2550659, 583.2550659
3: -154.5595398, 561.9185181, -154.5595398, 561.9185181, -716.4779663, 716.4779663
4: -131.8280182, 512.9211426, -131.8280182, 512.9211426, -644.7491455, 644.7491455

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9646884, upper bound: 554.9517015
time: 1.42 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9495942, upper bound: 554.9495942
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -137.6115265, 437.8572693, -115.3944626, 363.5161438, -501.1276550, 553.2517090
1: -193.6290741, 440.6151428, -161.7974091, 366.7948303, -560.4238892, 602.4124146
2: -163.5594940, 487.1738281, -136.8181305, 405.7499084, -569.3093872, 623.9919434
3: -172.4905701, 626.0422974, -144.2350464, 521.4130859, -693.9035034, 770.2773438
4: -146.9165649, 572.0592651, -123.1167984, 476.9207153, -623.8372803, 695.1760864

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9534835, upper bound: 554.9489196
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9584118, upper bound: 554.9717693
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9789981, upper bound: 554.9771857
time: 1.20 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -137.6115265, 437.8572693, -123.2541962, 392.5810242, -530.1925659, 561.1114502
1: -193.6290741, 440.6151428, -173.5091553, 394.9635315, -588.5925293, 614.1242676
2: -163.5594940, 487.1738281, -146.6661682, 436.5891418, -600.1486206, 633.8397827
3: -172.4905701, 626.0422974, -154.5595398, 561.9185181, -734.4088745, 780.6018066
4: -146.9165649, 572.0592651, -131.8280182, 512.9211426, -659.8377075, 703.8872681

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9534835, upper bound: 554.9489196
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9309788, upper bound: 554.9485551
time: 1.18 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9794499, upper bound: 554.9789849
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -123.2539520, 392.5803528, -131.5360107, 414.6815186, -537.9354858, 524.1163330
1: -173.5087891, 394.9628601, -184.4300995, 418.1284180, -591.6371460, 579.3929443
2: -146.6658630, 436.5884094, -155.8180084, 462.7027588, -609.3686523, 592.4064331
3: -154.5592194, 561.9174805, -164.3947754, 593.6577148, -748.2167969, 726.3121948
4: -131.8277740, 512.9202881, -140.0432434, 543.4032593, -675.2310181, 652.9635010

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9728163, upper bound: 554.9750471
time: 1.18 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9717595, upper bound: 554.9782593
time: 1.36 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9771857, upper bound: 554.9789981
time: 1.16 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -123.2539520, 392.5803528, -137.6115265, 437.8572693, -561.1112061, 530.1918945
1: -173.5087891, 394.9628601, -193.6290741, 440.6151428, -614.1239014, 588.5918579
2: -146.6658630, 436.5884094, -163.5594940, 487.1738281, -633.8396606, 600.1478882
3: -154.5592194, 561.9174805, -172.4905701, 626.0422974, -780.6015015, 734.4078979
4: -131.8277740, 512.9202881, -146.9165649, 572.0592651, -703.8870239, 659.8368530

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9728163, upper bound: 554.9750471
time: 1.18 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9717595, upper bound: 554.9782593
time: 1.26 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9771857, upper bound: 554.9789981
time: 1.59 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -137.6115265, 437.8572693, -131.5360107, 414.6815186, -552.2930298, 569.3933105
1: -193.6290741, 440.6151428, -184.4300995, 418.1284180, -611.7573242, 625.0452271
2: -163.5594940, 487.1738281, -155.8180084, 462.7027588, -626.2622681, 642.9918213
3: -172.4905701, 626.0422974, -164.3947754, 593.6577148, -766.1481323, 790.4370728
4: -146.9165649, 572.0592651, -140.0432434, 543.4032593, -690.3198242, 712.1025391

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9584118, upper bound: 554.9717693
time: 1.55 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9789981, upper bound: 554.9784832
time: 1.41 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -137.6115265, 437.8572693, -137.6115265, 437.8572693, -575.4688110, 575.4688110
1: -193.6290741, 440.6151428, -193.6290741, 440.6151428, -634.2441406, 634.2441406
2: -163.5594940, 487.1738281, -163.5594940, 487.1738281, -650.7333374, 650.7333374
3: -172.4905701, 626.0422974, -172.4905701, 626.0422974, -798.5327759, 798.5327759
4: -146.9165649, 572.0592651, -146.9165649, 572.0592651, -718.9758301, 718.9758301

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9309788, upper bound: 554.9485551
time: 1.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9794499, upper bound: 554.9790678
time: 1.18 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 6.39 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9686706, upper bound: 554.9541939
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9693960, upper bound: 554.9567523
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9686706, upper bound: 554.9541939
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9693960, upper bound: 554.9567523
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9828438
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9855075, upper bound: 554.9825508
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9828438
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9855075, upper bound: 554.9825508
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9839967
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9855023, upper bound: 554.9831879
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9839967
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9855023, upper bound: 554.9831879
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9829312
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9855044, upper bound: 554.9826638
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9855282, upper bound: 554.9829312
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9855044, upper bound: 554.9826638
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9646884, upper bound: 554.9517015
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9495942, upper bound: 554.9495942
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9646884, upper bound: 554.9517015
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9495942, upper bound: 554.9495942
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9584118, upper bound: 554.9717693
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9789981, upper bound: 554.9771857
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9309788, upper bound: 554.9485551
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9794499, upper bound: 554.9789849
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9717595, upper bound: 554.9782593
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9771857, upper bound: 554.9789981
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9717595, upper bound: 554.9782593
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9771857, upper bound: 554.9789981
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9584118, upper bound: 554.9717693
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9789981, upper bound: 554.9784832
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9309788, upper bound: 554.9485551
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.39
Output dim: 0, lower bound: -554.9794499, upper bound: 554.9790678

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -107.9422073, 340.9052429, -113.6845932, 358.3291626, -466.2713623, 454.5898438
1: -151.1541901, 344.0680847, -159.3527527, 361.5893555, -512.7434692, 503.4207764
2: -127.8792114, 380.6850891, -134.7651978, 400.0075989, -527.8867798, 515.4501343
3: -134.8178406, 489.5430298, -142.0730896, 514.1177979, -648.9356689, 631.6160889
4: -115.1885223, 447.4972839, -121.2956543, 470.1802979, -585.3688354, 568.7929077

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9847541, upper bound: 554.9847541
time: 1.35 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9847541, upper bound: 554.9860230
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -115.9024429, 365.3919067, -114.8263779, 361.6081848, -477.5106201, 480.2182617
1: -162.3976135, 368.5887756, -160.9953156, 364.8996277, -527.2972412, 529.5839844
2: -137.2470093, 407.8344116, -136.1293335, 403.6561584, -540.9031982, 543.9637451
3: -144.7448730, 524.3020020, -143.5135193, 518.6928711, -663.4376831, 667.8155518
4: -123.4825363, 479.4687805, -122.4991989, 474.4530945, -597.9356079, 601.9679565

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9860230, upper bound: 554.9871704
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9860230, upper bound: 554.9887241
time: 1.25 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -107.9422073, 340.9052429, -121.6398315, 387.6501770, -495.5923767, 462.5450439
1: -151.1541901, 344.0680847, -171.1968536, 389.9920349, -541.1461182, 515.2648315
2: -127.8792114, 380.6850891, -144.7288971, 431.1314392, -559.0106201, 525.4139404
3: -134.8178406, 489.5430298, -152.5183563, 554.9469604, -689.7647705, 642.0613403
4: -115.1885223, 447.4972839, -130.1130829, 506.4978333, -621.6863403, 577.6103516

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9686706, upper bound: 554.9541939
time: 1.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9686706, upper bound: 554.9541939
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -115.9024429, 365.3919067, -122.4074173, 389.8308105, -505.7332458, 487.7992554
1: -162.3976135, 368.5887756, -172.3255768, 392.2160339, -554.6135254, 540.9142456
2: -137.2470093, 407.8344116, -145.6520844, 433.5452881, -570.7922974, 553.4863892
3: -144.7448730, 524.3020020, -153.4980621, 557.9827881, -702.7276001, 677.8000488
4: -123.4825363, 479.4687805, -130.9218140, 509.3380737, -632.8206177, 610.3906250

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9693960, upper bound: 554.9567523
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9693960, upper bound: 554.9567523
time: 1.27 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -123.8264008, 391.1094055, -114.3501434, 359.9737854, -483.8001709, 505.4595337
1: -173.4983215, 394.4565735, -160.3298187, 363.3033142, -536.8015747, 554.7863770
2: -146.6014099, 436.4610291, -135.5867615, 401.8574829, -548.4588013, 572.0476074
3: -154.7384491, 560.1813354, -142.9323273, 516.3413086, -671.0797729, 703.1135864
4: -131.8179932, 512.5811157, -122.0218277, 472.3112488, -604.1292725, 634.6029663

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9860230, upper bound: 554.9858865
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9870658, upper bound: 554.9867459
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -129.3510437, 407.1157227, -114.8304596, 361.5578308, -490.9088440, 521.9461670
1: -181.2432861, 410.7232971, -160.9791107, 364.8809814, -546.1240845, 571.7023926
2: -153.1453857, 454.5035400, -136.1311340, 403.6305237, -556.7758789, 590.6346436
3: -161.5780792, 583.0626831, -143.5103149, 518.6823120, -680.2603760, 726.5729980
4: -137.6862946, 533.7856445, -122.5088654, 474.4376831, -612.1239624, 656.2943726

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855321, upper bound: 554.9854938
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9865347, upper bound: 554.9862808
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -123.8264008, 391.1094055, -122.3447800, 389.5955811, -513.4219971, 513.4541016
1: -173.4983215, 394.4565735, -172.2271729, 391.9877930, -565.4860840, 566.6837158
2: -146.6014099, 436.4610291, -145.5876770, 433.2721863, -579.8735962, 582.0485840
3: -154.7384491, 560.1813354, -153.4226837, 557.6222534, -712.3607178, 713.6038818
4: -131.8179932, 512.5811157, -130.8690643, 508.9939575, -640.8119507, 643.4501953

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9822411, upper bound: 554.9778067
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.8084294, upper bound: 554.9024946
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9828233, upper bound: 554.9753523
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9851220, upper bound: 554.9812639
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -129.3510437, 407.1157227, -122.6289215, 390.3940735, -519.7451172, 529.7446289
1: -181.2432861, 410.7232971, -172.6064911, 392.8145447, -574.0577393, 583.3297729
2: -153.1453857, 454.5035400, -145.9085236, 434.2135315, -587.3588867, 600.4120483
3: -161.5780792, 583.0626831, -153.7590485, 558.8358765, -720.4138794, 736.8217163
4: -137.6862946, 533.7856445, -131.1551514, 510.1242981, -647.8105469, 664.9407959

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9819877, upper bound: 554.9776165
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9776243, upper bound: 554.9755399
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9829625, upper bound: 554.9752079
time: 1.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9851113, upper bound: 554.9809475
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -107.2945023, 338.7147522, -130.4687958, 411.0578308, -518.3521729, 469.1835327
1: -150.3399353, 341.8565674, -182.9297333, 414.5599670, -564.8999023, 524.7863159
2: -127.1170197, 378.1401367, -154.5604858, 458.7315979, -585.8486328, 532.7006226
3: -134.0518646, 486.1622314, -163.0618286, 588.4913330, -722.5431519, 649.2240601
4: -114.4648819, 444.4047546, -138.9237671, 538.7106934, -653.1755981, 583.3283691

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9860230, upper bound: 554.9872889
time: 1.49 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9867333, upper bound: 554.9878124
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -113.1604538, 355.7814636, -130.9788513, 412.7480774, -525.9085083, 486.7603149
1: -158.5563812, 359.2316284, -183.6178894, 416.2369385, -574.7933350, 542.8494263
2: -134.0978394, 397.3772888, -155.1371613, 460.6079102, -594.7057495, 552.5144653
3: -141.3645935, 510.6240845, -163.6766205, 590.9512329, -732.3157959, 674.3007202
4: -120.7083435, 467.1063843, -139.4426880, 540.9466553, -661.6550293, 606.5490723

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855418, upper bound: 554.9859908
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9862808, upper bound: 554.9865347
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -107.2945023, 338.7147522, -136.6123352, 434.5274353, -541.8219604, 475.3270874
1: -150.3399353, 341.8565674, -192.2177734, 437.3170471, -587.6569824, 534.0743408
2: -127.1170197, 378.1401367, -162.3764801, 483.5070190, -610.6240234, 540.5166016
3: -134.0518646, 486.1622314, -171.2389526, 621.2889404, -755.3407593, 657.4011230
4: -114.4648819, 444.4047546, -145.8619690, 567.7225952, -682.1875000, 590.2666626

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9822411, upper bound: 554.9792325
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9726515, upper bound: 554.9606934
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9851029, upper bound: 554.9836038
time: 1.34 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -113.1604538, 355.7814636, -137.0016785, 435.7284546, -548.8889160, 492.7831421
1: -158.5563812, 359.2316284, -192.7456207, 438.5259399, -597.0822144, 551.9772339
2: -134.0978394, 397.3772888, -162.8183746, 484.8620605, -618.9598389, 560.1956177
3: -141.3645935, 510.6240845, -171.7086029, 623.0444336, -764.4089966, 682.3327026
4: -120.7083435, 467.1063843, -146.2606201, 569.3338013, -690.0421143, 613.3670044

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9819663, upper bound: 554.9782362
time: 1.26 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9735407, upper bound: 554.9606561
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9850385, upper bound: 554.9828836
time: 1.36 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -123.8264008, 391.1094055, -130.4687958, 411.0578308, -534.8840942, 521.5780640
1: -173.4983215, 394.4565735, -182.9297333, 414.5599670, -588.0582275, 577.3862915
2: -146.6014099, 436.4610291, -154.5604858, 458.7315979, -605.3330078, 591.0213623
3: -154.7384491, 560.1813354, -163.0618286, 588.4913330, -743.2297974, 723.2431641
4: -131.8179932, 512.5811157, -138.9237671, 538.7106934, -670.5286865, 651.5047607

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9860230, upper bound: 554.9858865
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9867333, upper bound: 554.9867459
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -129.3510437, 407.1157227, -130.9788513, 412.7480774, -542.0991211, 538.0946045
1: -181.2432861, 410.7232971, -183.6178894, 416.2369385, -597.4801025, 594.3411865
2: -153.1453857, 454.5035400, -155.1371613, 460.6079102, -613.7532959, 609.6406860
3: -161.5780792, 583.0626831, -163.6766205, 590.9512329, -752.5291748, 746.7393188
4: -137.6862946, 533.7856445, -139.4426880, 540.9466553, -678.6328735, 673.2283325

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855321, upper bound: 554.9854938
time: 1.24 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9862808, upper bound: 554.9862808
time: 4.02 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -123.8264008, 391.1094055, -136.6123352, 434.5274353, -558.3538208, 527.7216187
1: -173.4983215, 394.4565735, -192.2177734, 437.3170471, -610.8153687, 586.6743164
2: -146.6014099, 436.4610291, -162.3764801, 483.5070190, -630.1083984, 598.8372803
3: -154.7384491, 560.1813354, -171.2389526, 621.2889404, -776.0274048, 731.4202271
4: -131.8179932, 512.5811157, -145.8619690, 567.7225952, -699.5405884, 658.4431152

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9822411, upper bound: 554.9778143
time: 1.39 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9712836, upper bound: 554.9719385
time: 1.25 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.8558769, upper bound: 554.8381752
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9851220, upper bound: 554.9821848
time: 1.22 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -129.3510437, 407.1157227, -137.0016785, 435.7284546, -565.0794678, 544.1174316
1: -181.2432861, 410.7232971, -192.7456207, 438.5259399, -619.7689819, 603.4689331
2: -153.1453857, 454.5035400, -162.8183746, 484.8620605, -638.0074463, 617.3218994
3: -161.5780792, 583.0626831, -171.7086029, 623.0444336, -784.6223145, 754.7713013
4: -137.6862946, 533.7856445, -146.2606201, 569.3338013, -707.0200806, 680.0462646

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9819663, upper bound: 554.9776303
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9778740, upper bound: 554.9755978
time: 1.26 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9734667, upper bound: 554.9598473
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9851139, upper bound: 554.9820845
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -127.4574890, 404.6058350, -113.2848587, 356.2744446, -483.7319031, 517.8906860
1: -179.1875305, 407.3471985, -158.7934723, 359.5983887, -538.7858887, 566.1405640
2: -151.3261414, 450.4775696, -134.2819977, 397.7825317, -549.1085815, 584.7595825
3: -159.6564789, 578.5159302, -141.5607452, 511.0424194, -670.6988525, 720.0765991
4: -135.9640045, 528.9044800, -120.8404083, 467.5661316, -603.5301514, 649.7448730

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9606934, upper bound: 554.9726515
time: 1.23 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9606561, upper bound: 554.9735407
time: 1.31 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -134.7304993, 428.3551331, -114.9845963, 362.1799011, -496.9104004, 543.3397217
1: -189.5167542, 431.1124573, -161.2124481, 365.4597168, -554.9764404, 592.3248901
2: -160.0974274, 476.7073669, -136.3240509, 404.2750244, -564.3724365, 613.0313721
3: -168.8433228, 612.4936523, -143.7158813, 519.5113525, -688.3546753, 756.2094116
4: -143.8236237, 559.7412109, -122.6747894, 475.1908875, -619.0144653, 682.4159546

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9786863, upper bound: 554.9773979
time: 1.11 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9836038, upper bound: 554.9851029
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9828836, upper bound: 554.9850385
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -135.2478180, 429.6503296, -122.6289215, 390.3940735, -525.6419067, 552.2791748
1: -190.2034760, 432.5535889, -172.6064911, 392.8145447, -583.0180054, 605.1600952
2: -160.6852112, 478.2824402, -145.9085236, 434.2135315, -594.8987427, 624.1909790
3: -169.4587860, 614.4909668, -153.7590485, 558.8358765, -728.2946777, 768.2500000
4: -144.3734741, 561.6165771, -131.1551514, 510.1242981, -654.4977417, 692.7717285

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9750471, upper bound: 554.9728163
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9782593, upper bound: 554.9717595
time: 1.86 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9789981, upper bound: 554.9771857
time: 3.29 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -112.6693802, 357.8695679, -129.5573883, 407.9788513, -520.6482544, 487.4269409
1: -158.3960114, 360.2498779, -181.6186371, 411.4453430, -569.8411255, 541.8685303
2: -133.8866272, 398.2899475, -153.4425659, 455.3129578, -589.1994629, 551.7325439
3: -141.1417084, 512.4230957, -161.8929749, 584.0025024, -725.1442261, 674.3160400
4: -120.4153976, 467.9117432, -137.9091797, 534.7083130, -655.1236572, 605.8209229

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9130079, upper bound: 554.9318248
time: 1.24 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9753523, upper bound: 554.9828233
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9752079, upper bound: 554.9829625
time: 1.26 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -120.6352081, 383.9923096, -131.0917206, 413.2080688, -533.8432617, 515.0839844
1: -169.7540283, 386.3453674, -183.7973480, 416.6609192, -586.4149170, 570.1426392
2: -143.5063934, 427.1207581, -155.2845306, 461.0799866, -604.5863647, 582.4052124
3: -151.2342529, 549.6581421, -163.8322144, 591.5593262, -742.7935791, 713.4903564
4: -129.0052338, 501.7810059, -139.5665588, 541.5011597, -670.5063477, 641.3474731

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9812639, upper bound: 554.9851220
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9809475, upper bound: 554.9851113
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -112.6693802, 357.8695679, -135.5733795, 430.8937683, -543.5631714, 493.4429321
1: -158.3960114, 360.2498779, -190.7261810, 433.6620178, -592.0577393, 550.9760742
2: -133.8866272, 398.2899475, -161.1080627, 479.5069275, -613.3934326, 559.3979492
3: -141.1417084, 512.4230957, -169.9134216, 616.0382080, -757.1799316, 682.3365479
4: -120.4153976, 467.9117432, -144.7123260, 563.0415039, -683.4569092, 612.6240845

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9130079, upper bound: 554.9318248
time: 1.24 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9243957, upper bound: 554.9246001
time: 1.06 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9717595, upper bound: 554.9782593
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -120.6352081, 383.9923096, -137.1773224, 436.4225159, -557.0576782, 521.1696167
1: -169.7540283, 386.3453674, -193.0095978, 439.1800842, -608.9340820, 579.3549805
2: -143.5063934, 427.1207581, -163.0377655, 485.5923767, -629.0987549, 590.1585083
3: -151.2342529, 549.6581421, -171.9410706, 623.9941406, -775.2283936, 721.5992432
4: -129.0052338, 501.7810059, -146.4499664, 570.1991577, -699.2043457, 648.2309570

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9476795, upper bound: 554.9300148
time: 1.22 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9771857, upper bound: 554.9789981
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -127.4574890, 404.6058350, -129.5573883, 407.9788513, -535.4362793, 534.1632080
1: -179.1875305, 407.3471985, -181.6186371, 411.4453430, -590.6327515, 588.9658203
2: -151.3261414, 450.4775696, -153.4425659, 455.3129578, -606.6389771, 603.9201660
3: -159.6564789, 578.5159302, -161.8929749, 584.0025024, -743.6588745, 740.4089355
4: -135.9640045, 528.9044800, -137.9091797, 534.7083130, -670.6721802, 666.8136597

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.8381752, upper bound: 554.8558769
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9598473, upper bound: 554.9734667
time: 1.30 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -134.7304993, 428.3551331, -131.0917206, 413.2080688, -547.9385986, 559.4467773
1: -189.5167542, 431.1124573, -183.7973480, 416.6609192, -606.1776733, 614.9097900
2: -160.0974274, 476.7073669, -155.2845306, 461.0799866, -621.1774292, 631.9918213
3: -168.8433228, 612.4936523, -163.8322144, 591.5593262, -760.4026489, 776.3258057
4: -143.8236237, 559.7412109, -139.5665588, 541.5011597, -685.3247681, 699.3076782

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9786863, upper bound: 554.9772053
time: 1.22 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9831669, upper bound: 554.9851029
time: 1.13 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9828209, upper bound: 554.9850427
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -135.2478180, 429.6503296, -137.0016785, 435.7284546, -570.9762573, 566.6519165
1: -190.2034760, 432.5535889, -192.7456207, 438.5259399, -628.7291870, 625.2991943
2: -160.6852112, 478.2824402, -162.8183746, 484.8620605, -645.5472412, 641.1007690
3: -169.4587860, 614.4909668, -171.7086029, 623.0444336, -792.5031738, 786.1995850
4: -144.3734741, 561.6165771, -146.2606201, 569.3338013, -713.7072754, 707.8771973

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9751379, upper bound: 554.9728163
time: 1.54 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9743172, upper bound: 554.9726771
time: 1.37 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.96 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9847541, upper bound: 554.9847541
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9847541, upper bound: 554.9860230
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9860230, upper bound: 554.9871704
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9860230, upper bound: 554.9887241
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9686706, upper bound: 554.9541939
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9686706, upper bound: 554.9541939
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9693960, upper bound: 554.9567523
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9693960, upper bound: 554.9567523
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9860230, upper bound: 554.9858865
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9870658, upper bound: 554.9867459
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9855321, upper bound: 554.9854938
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9865347, upper bound: 554.9862808
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9828233, upper bound: 554.9753523
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9851220, upper bound: 554.9812639
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9829625, upper bound: 554.9752079
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9851113, upper bound: 554.9809475
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9860230, upper bound: 554.9872889
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9867333, upper bound: 554.9878124
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9855418, upper bound: 554.9859908
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9862808, upper bound: 554.9865347
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9726515, upper bound: 554.9606934
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9851029, upper bound: 554.9836038
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9735407, upper bound: 554.9606561
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9850385, upper bound: 554.9828836
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9860230, upper bound: 554.9858865
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9867333, upper bound: 554.9867459
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9855321, upper bound: 554.9854938
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9862808, upper bound: 554.9862808
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.8558769, upper bound: 554.8381752
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9851220, upper bound: 554.9821848
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9734667, upper bound: 554.9598473
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9851139, upper bound: 554.9820845
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9606934, upper bound: 554.9726515
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9606561, upper bound: 554.9735407
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9836038, upper bound: 554.9851029
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9828836, upper bound: 554.9850385
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9782593, upper bound: 554.9717595
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9789981, upper bound: 554.9771857
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9753523, upper bound: 554.9828233
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9752079, upper bound: 554.9829625
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9812639, upper bound: 554.9851220
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9809475, upper bound: 554.9851113
IS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9243957, upper bound: 554.9246001
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9717595, upper bound: 554.9782593
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9476795, upper bound: 554.9300148
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9771857, upper bound: 554.9789981
IS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.8381752, upper bound: 554.8558769
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9598473, upper bound: 554.9734667
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9831669, upper bound: 554.9851029
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9828209, upper bound: 554.9850427
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9751379, upper bound: 554.9728163
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 0, lower bound: -554.9743172, upper bound: 554.9726771

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -107.9422073, 340.9052429, -107.9422073, 340.9052429, -448.8474426, 448.8474426
1: -151.1541901, 344.0680847, -151.1541901, 344.0680847, -495.2222595, 495.2222595
2: -127.8792114, 380.6850891, -127.8792114, 380.6850891, -508.5643005, 508.5643005
3: -134.8178406, 489.5430298, -134.8178406, 489.5430298, -624.3608398, 624.3608398
4: -115.1885223, 447.4972839, -115.1885223, 447.4972839, -562.6857910, 562.6857910

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9839905, upper bound: 554.9844395
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9847541, upper bound: 554.9847541
time: 1.31 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -107.9422073, 340.9052429, -115.9024429, 365.3919067, -473.3341064, 456.8076782
1: -151.1541901, 344.0680847, -162.3976135, 368.5887756, -519.7427979, 506.4656677
2: -127.8792114, 380.6850891, -137.2470093, 407.8344116, -535.7135620, 517.9319458
3: -134.8178406, 489.5430298, -144.7448730, 524.3020020, -659.1198730, 634.2878418
4: -115.1885223, 447.4972839, -123.4825363, 479.4687805, -594.6572876, 570.9797974

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9839905, upper bound: 554.9851960
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9847541, upper bound: 554.9855321
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -115.9024429, 365.3919067, -107.9422073, 340.9052429, -456.8076782, 473.3340759
1: -162.3976135, 368.5887756, -151.1541901, 344.0680847, -506.4656677, 519.7427979
2: -137.2470093, 407.8344116, -127.8792114, 380.6850891, -517.9319458, 535.7135620
3: -144.7448730, 524.3020020, -134.8178406, 489.5430298, -634.2878418, 659.1198730
4: -123.4825363, 479.4687805, -115.1885223, 447.4972839, -570.9797974, 594.6572876

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9860230, upper bound: 554.9871704
time: 1.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855321, upper bound: 554.9857900
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -115.9024429, 365.3919067, -115.9024429, 365.3919067, -481.2943420, 481.2943420
1: -162.3976135, 368.5887756, -162.3976135, 368.5887756, -530.9862671, 530.9863281
2: -137.2470093, 407.8344116, -137.2470093, 407.8344116, -545.0814209, 545.0814209
3: -144.7448730, 524.3020020, -144.7448730, 524.3020020, -669.0468750, 669.0468750
4: -123.4825363, 479.4687805, -123.4825363, 479.4687805, -602.9512939, 602.9512939

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9860230, upper bound: 554.9878124
time: 1.32 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855321, upper bound: 554.9865347
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -107.9422073, 340.9052429, -116.3693695, 371.5358887, -479.4780884, 457.2745972
1: -151.1541901, 344.0680847, -163.6532745, 373.7986450, -524.9526367, 507.7213745
2: -127.8792114, 380.6850891, -138.4061279, 413.3215332, -541.2007446, 519.0911255
3: -134.8178406, 489.5430298, -145.8605499, 532.1776733, -666.9954834, 635.4035645
4: -115.1885223, 447.4972839, -124.5224457, 485.6073914, -600.7958374, 572.0196533

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9593675, upper bound: 554.9493596
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9686706, upper bound: 554.9541939
time: 1.41 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -107.9422073, 340.9052429, -121.5298615, 387.2480164, -495.1902161, 462.4350586
1: -151.1541901, 344.0680847, -171.0144043, 389.6676025, -540.8217163, 515.0825195
2: -127.8792114, 380.6850891, -144.4824677, 430.9747925, -558.8540039, 525.1672974
3: -134.8178406, 489.5430298, -152.3258514, 554.6708984, -689.4887695, 641.8688965
4: -115.1885223, 447.4972839, -129.8988190, 506.4743652, -621.6629028, 577.3961182

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9593675, upper bound: 554.9493596
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9686706, upper bound: 554.9541939
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -115.9024429, 365.3919067, -116.3693695, 371.5358887, -487.4383240, 481.7612610
1: -162.3976135, 368.5887756, -163.6532745, 373.7986450, -536.1961670, 532.2419434
2: -137.2470093, 407.8344116, -138.4061279, 413.3215332, -550.5685425, 546.2405396
3: -144.7448730, 524.3020020, -145.8605499, 532.1776733, -676.9223633, 670.1625366
4: -123.4825363, 479.4687805, -124.5224457, 485.6073914, -609.0898438, 603.9912109

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9691216, upper bound: 554.9567523
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9693960, upper bound: 554.9554891
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -115.9024429, 365.3919067, -121.5298615, 387.2480164, -503.1504517, 486.9216919
1: -162.3976135, 368.5887756, -171.0144043, 389.6676025, -552.0651855, 539.6030884
2: -137.2470093, 407.8344116, -144.4824677, 430.9747925, -568.2218018, 552.3167725
3: -144.7448730, 524.3020020, -152.3258514, 554.6708984, -699.4157715, 676.6278687
4: -123.4825363, 479.4687805, -129.8988190, 506.4743652, -629.9569092, 609.3676147

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9691216, upper bound: 554.9567523
time: 1.36 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9693960, upper bound: 554.9554891
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -122.2211914, 386.2478943, -106.8881836, 337.3396606, -459.5607605, 493.1360779
1: -171.2013855, 389.5922546, -149.6711273, 340.5508423, -511.7522278, 539.2632446
2: -144.6759949, 431.0885315, -126.6347885, 376.7598877, -521.4357910, 557.7233276
3: -152.7127533, 553.3631592, -133.5027161, 484.4366760, -637.1494141, 686.8658447
4: -130.1155090, 506.2788086, -114.0819626, 442.8493652, -572.9648438, 620.3607788

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9850153, upper bound: 554.9847154
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9850153, upper bound: 554.9858865
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -123.0517273, 388.5735779, -114.7702103, 361.5015259, -484.5532532, 503.3437805
1: -172.4101410, 391.9309387, -160.7977753, 364.7720032, -537.1820068, 552.7286987
2: -145.6686401, 433.6704712, -135.9079132, 403.5805969, -549.2492676, 569.5783691
3: -153.7595673, 556.5906982, -143.3240814, 518.7937012, -672.5532837, 699.9146729
4: -130.9845428, 509.3041992, -122.2915039, 474.4503174, -605.4348145, 631.5957031

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9859908, upper bound: 554.9855280
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9859908, upper bound: 554.9867459
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -127.6632309, 402.0028381, -107.3716354, 338.9287109, -466.5919189, 509.3744812
1: -178.8398590, 405.5960999, -150.3273926, 342.1293335, -520.9691162, 555.9234619
2: -151.1246490, 448.8402710, -127.1853943, 378.5389099, -529.6634521, 576.0256348
3: -159.4537659, 575.8593140, -134.0851288, 486.7737732, -646.2274780, 709.9443970
4: -135.8967133, 527.1468506, -114.5736313, 444.9797058, -580.8763428, 641.7204590

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9850153, upper bound: 554.9847553
time: 1.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9850153, upper bound: 554.9854938
time: 1.26 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -128.6597290, 404.8434448, -115.3690109, 363.5663147, -492.2259827, 520.2124634
1: -180.2585754, 408.4558411, -161.6251984, 366.7919312, -547.0504150, 570.0810547
2: -152.3112640, 451.9981079, -136.5974884, 405.8448181, -558.1560669, 588.5955811
3: -160.7017212, 579.8182983, -144.0617828, 521.7334595, -682.4351807, 723.8800659
4: -136.9373627, 530.8334351, -122.9076767, 477.1324158, -614.0697632, 653.7410889

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9857900, upper bound: 554.9855418
time: 1.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9859908, upper bound: 554.9862808
time: 1.30 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -121.7854080, 384.1528931, -111.6031799, 354.3011169, -476.0865173, 495.7560730
1: -170.5970917, 387.5322876, -156.8861389, 356.7179260, -527.3149414, 544.4184570
2: -144.1423950, 428.8181152, -132.6186676, 394.3619995, -538.5043335, 561.4367676
3: -152.1510925, 550.2003784, -139.8018646, 507.3354492, -659.4864502, 690.0022583
4: -129.6111450, 503.5742493, -119.2866898, 463.2633057, -592.8743286, 622.8608398

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9557272, upper bound: 554.9646176
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9782279, upper bound: 554.9717274
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -123.3788147, 389.6261597, -119.7221832, 380.9803467, -504.3591309, 509.3483276
1: -172.8604279, 392.9817810, -168.4651337, 383.3404846, -556.2008667, 561.4468994
2: -146.0642853, 434.8309326, -142.4222260, 423.7734680, -569.8377686, 577.2531738
3: -154.1714630, 558.0779419, -150.0909271, 545.3186035, -699.4900513, 708.1688843
4: -131.3381348, 510.6707153, -128.0404968, 497.8161621, -629.1542969, 638.7111816

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9812933, upper bound: 554.9772190
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9782279, upper bound: 554.9772261
time: 1.35 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -127.3302689, 400.2829590, -112.0754395, 355.8074951, -483.1377563, 512.3583984
1: -178.3710480, 403.9018555, -157.5387421, 358.2322693, -536.6033325, 561.4406128
2: -150.7203217, 446.9702759, -133.1673889, 396.0696411, -546.7897949, 580.1376953
3: -159.0233612, 573.2145386, -140.3824463, 509.5194702, -668.5428467, 713.5968628
4: -135.5091858, 524.9152222, -119.7774582, 465.3097534, -600.8189697, 644.6926270

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9318248, upper bound: 554.9130079
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9782955, upper bound: 554.9710493
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9782279, upper bound: 554.9715891
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -128.9060364, 405.6426086, -119.9828033, 381.7068176, -510.6128540, 525.6252441
1: -180.6098785, 409.2561035, -168.8115845, 384.1046448, -564.7143555, 578.0676880
2: -152.6113434, 452.8811951, -142.7147980, 424.6438904, -577.2552490, 595.5960083
3: -161.0149384, 580.9649658, -150.3988037, 546.4433594, -707.4581909, 731.3637695
4: -137.2091827, 531.8845215, -128.3030701, 498.8815918, -636.0907593, 660.1876221

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9812820, upper bound: 554.9770704
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9794360, upper bound: 554.9766647
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -105.5884705, 333.5523071, -123.1939850, 389.0049133, -494.5933228, 456.7462769
1: -147.8912201, 336.6539307, -172.5678864, 392.4465942, -540.3378296, 509.2218018
2: -125.0628357, 372.3996277, -145.8530121, 434.3125000, -559.3753052, 518.2525635
3: -131.8903503, 478.8645325, -153.9025726, 557.4547119, -689.3450928, 632.7670898
4: -112.6438599, 437.6669922, -131.2149048, 510.0990295, -622.7428589, 568.8818970

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9839905, upper bound: 554.9846791
time: 1.25 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9839905, upper bound: 554.9872889
time: 1.47 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.09 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9839905, upper bound: 554.9844395
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9847541, upper bound: 554.9847541
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9839905, upper bound: 554.9851960
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9847541, upper bound: 554.9855321
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9860230, upper bound: 554.9871704
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9855321, upper bound: 554.9857900
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9860230, upper bound: 554.9878124
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9855321, upper bound: 554.9865347
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9593675, upper bound: 554.9493596
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9686706, upper bound: 554.9541939
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9593675, upper bound: 554.9493596
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9686706, upper bound: 554.9541939
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9691216, upper bound: 554.9567523
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9693960, upper bound: 554.9554891
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9691216, upper bound: 554.9567523
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9693960, upper bound: 554.9554891
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9850153, upper bound: 554.9847154
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9850153, upper bound: 554.9858865
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9859908, upper bound: 554.9855280
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9859908, upper bound: 554.9867459
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9850153, upper bound: 554.9847553
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9850153, upper bound: 554.9854938
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9857900, upper bound: 554.9855418
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9859908, upper bound: 554.9862808
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9557272, upper bound: 554.9646176
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9782279, upper bound: 554.9717274
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9812933, upper bound: 554.9772190
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9782279, upper bound: 554.9772261
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9782955, upper bound: 554.9710493
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9782279, upper bound: 554.9715891
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9812820, upper bound: 554.9770704
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9794360, upper bound: 554.9766647
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9839905, upper bound: 554.9846791
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.09
Output dim: 0, lower bound: -554.9839905, upper bound: 554.9872889
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9867333, upper bound: 554.9878124
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9855418, upper bound: 554.9859908
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9862808, upper bound: 554.9865347
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9726515, upper bound: 554.9606934
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9851029, upper bound: 554.9836038
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9735407, upper bound: 554.9606561
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9850385, upper bound: 554.9828836
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9860230, upper bound: 554.9858865
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9867333, upper bound: 554.9867459
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9855321, upper bound: 554.9854938
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9862808, upper bound: 554.9862808
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9851220, upper bound: 554.9821848
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9734667, upper bound: 554.9598473
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9851139, upper bound: 554.9820845
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9606934, upper bound: 554.9726515
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9606561, upper bound: 554.9735407
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9836038, upper bound: 554.9851029
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9828836, upper bound: 554.9850385
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9782593, upper bound: 554.9717595
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9789981, upper bound: 554.9771857
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9753523, upper bound: 554.9828233
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9752079, upper bound: 554.9829625
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9812639, upper bound: 554.9851220
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9809475, upper bound: 554.9851113
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9717595, upper bound: 554.9782593
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9771857, upper bound: 554.9789981
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9598473, upper bound: 554.9734667
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9831669, upper bound: 554.9851029
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9828209, upper bound: 554.9850427
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9751379, upper bound: 554.9728163
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -554.9743172, upper bound: 554.9726771
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=619.6494140625
rel_dist={0: [-554.9907236424904, 554.9907236424906]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9822572, upper bound: 554.9821983
time: 1.09 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9788198, upper bound: 554.9788198
time: 1.00 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.27 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.27
Output dim: 0, lower bound: -554.9822572, upper bound: 554.9821983
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.27
Output dim: 0, lower bound: -554.9788198, upper bound: 554.9788198

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -135.4273682, 427.6317749, -146.1165924, 464.6019897, -600.0292969, 573.7483521
1: -189.9990234, 430.8988953, -205.5300140, 467.4161682, -657.4151611, 636.4288330
2: -160.4833527, 476.7970886, -173.5906677, 516.8294678, -677.3128052, 650.3877563
3: -169.3506470, 612.0695801, -183.1269836, 664.0928345, -833.4434204, 795.1965332
4: -144.2209015, 559.9951782, -155.9299164, 606.8745117, -751.0953979, 715.9249878

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9821432, upper bound: 554.9807618
time: 1.16 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9821432, upper bound: 554.9820893
time: 5.74 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -141.9587555, 452.3752747, -144.5160522, 460.2228699, -602.1816406, 596.8913574
1: -199.8379517, 454.9771729, -203.3786163, 462.9060364, -662.7439575, 658.3557739
2: -168.7648926, 503.0252075, -171.7580566, 511.8190002, -680.5836792, 674.7832031
3: -178.0149994, 646.6646729, -181.1892548, 657.8613281, -835.8763428, 827.8539429
4: -151.5778503, 590.7100830, -154.2735901, 601.0419922, -752.6198120, 744.9836426

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9788198, upper bound: 554.9780110
time: 1.27 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9787101, upper bound: 554.9787101
time: 1.32 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 5.01 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.01
Output dim: 0, lower bound: -554.9821432, upper bound: 554.9807618
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.01
Output dim: 0, lower bound: -554.9821432, upper bound: 554.9820893
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.01
Output dim: 0, lower bound: -554.9788198, upper bound: 554.9780110
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.01
Output dim: 0, lower bound: -554.9787101, upper bound: 554.9787101

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -131.9953461, 416.2260437, -127.0035095, 403.5899048, -535.5852661, 543.2293701
1: -185.1824799, 419.5390625, -178.6663513, 406.1940002, -591.3764648, 598.2053833
2: -156.4597168, 464.1816711, -151.0319366, 448.9767456, -605.4364624, 615.2136230
3: -165.0520630, 595.7461548, -159.1953735, 577.6365967, -742.6885376, 754.9415283
4: -140.6274719, 545.1712036, -135.7885284, 527.4961548, -668.1236572, 680.9596558

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9821432, upper bound: 554.9807618
time: 1.32 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9821432, upper bound: 554.9807618
time: 1.12 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -132.9245148, 419.2939148, -141.7274933, 449.9015808, -582.8259277, 561.0213013
1: -186.4261322, 422.6757812, -199.2690277, 452.8510132, -639.2771606, 621.9447632
2: -157.4899597, 467.7318420, -168.3435516, 500.7761841, -658.2659912, 636.0753784
3: -166.1679077, 600.2090454, -177.5541840, 643.1870728, -809.3548584, 777.7631226
4: -141.5402679, 549.3275146, -151.2301025, 587.9937744, -729.5339355, 700.5575562

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9767678, upper bound: 554.9771653
time: 1.18 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9759107, upper bound: 554.9771138
time: 1.07 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -139.1173553, 442.9570312, -125.6301498, 399.8784485, -538.9957886, 568.5871582
1: -195.8641663, 445.5037537, -176.8060608, 402.3511658, -598.2153320, 622.3098145
2: -165.4295807, 492.5388489, -149.4530487, 444.7727051, -610.2022705, 641.9917603
3: -174.4618378, 633.0667725, -157.5191956, 572.3540649, -746.8158569, 790.5859375
4: -148.5777740, 578.3828125, -134.3448944, 522.5493774, -671.1271362, 712.7276611

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9780110, upper bound: 554.9780110
time: 1.20 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9780110, upper bound: 554.9780110
time: 1.09 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -139.2904816, 443.4644775, -140.2022400, 445.8002319, -585.0906982, 583.6667480
1: -196.0285492, 446.1513062, -197.2199707, 448.6330261, -644.6615601, 643.3712769
2: -165.5709534, 493.2917786, -166.5955963, 496.0687561, -661.6397095, 659.8873901
3: -174.6245728, 633.9942627, -175.7086945, 637.3724365, -811.9969482, 809.7029419
4: -148.7182007, 579.2557983, -149.6502533, 582.5206299, -731.2387695, 728.9060059

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9722958, upper bound: 554.9740693
time: 1.62 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9740693, upper bound: 554.9740693
time: 1.43 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 5.92 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.92
Output dim: 0, lower bound: -554.9821432, upper bound: 554.9807618
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.92
Output dim: 0, lower bound: -554.9821432, upper bound: 554.9807618
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.92
Output dim: 0, lower bound: -554.9767678, upper bound: 554.9771653
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.92
Output dim: 0, lower bound: -554.9759107, upper bound: 554.9771138
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.92
Output dim: 0, lower bound: -554.9780110, upper bound: 554.9780110
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.92
Output dim: 0, lower bound: -554.9780110, upper bound: 554.9780110
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.92
Output dim: 0, lower bound: -554.9722958, upper bound: 554.9740693
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.92
Output dim: 0, lower bound: -554.9740693, upper bound: 554.9740693

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -115.3944626, 363.5161438, -127.0035095, 403.5899048, -518.9843140, 490.5196533
1: -161.7974091, 366.7948303, -178.6663513, 406.1940002, -567.9913940, 545.4611816
2: -136.8181305, 405.7499084, -151.0319366, 448.9767456, -585.7948608, 556.7818604
3: -144.2350464, 521.4130859, -159.1953735, 577.6365967, -721.8715820, 680.6083984
4: -123.1167984, 476.9207153, -135.7885284, 527.4961548, -650.6129150, 612.7092285

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9821432, upper bound: 554.9807618
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9821432, upper bound: 554.9807618
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -131.5360107, 414.6815186, -127.0035095, 403.5899048, -535.1259155, 541.6850586
1: -184.4300995, 418.1284180, -178.6663513, 406.1940002, -590.6240845, 596.7947388
2: -155.8180084, 462.7027588, -151.0319366, 448.9767456, -604.7947388, 613.7346802
3: -164.3947754, 593.6577148, -159.1953735, 577.6365967, -742.0313110, 752.8529663
4: -140.0432434, 543.4032593, -135.7885284, 527.4961548, -667.5393677, 679.1917725

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9821432, upper bound: 554.9807618
time: 1.32 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9821432, upper bound: 554.9807618
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -130.4546814, 411.3610229, -141.0437164, 447.6981506, -578.1527710, 552.4047241
1: -182.9963989, 414.6822205, -198.3174591, 450.6356812, -633.6320801, 612.9996948
2: -154.5989075, 458.8901978, -167.5439148, 498.3247986, -652.9237061, 626.4340820
3: -163.1041565, 588.7929077, -176.7060699, 640.0237427, -803.1279297, 765.4989624
4: -138.9424438, 538.9245605, -150.5117493, 585.1165161, -724.0588989, 689.4362793

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9732135, upper bound: 554.9720579
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9767678, upper bound: 554.9771653
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9767678, upper bound: 554.9771653
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -161.1584473, 510.1374817, -132.5584259, 419.2658081, -580.4242554, 642.6959229
1: -226.5817108, 514.0497437, -186.2056885, 422.6224976, -649.2041016, 700.2554321
2: -190.9772797, 568.9328613, -157.2342072, 467.5904236, -658.5676880, 726.1670532
3: -202.2574005, 730.7261353, -166.0338745, 600.1277466, -802.3851318, 896.7600098
4: -171.9734344, 669.8974609, -141.3705139, 549.2449341, -721.2182617, 811.2679443

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9746480, upper bound: 554.9771138
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9759107, upper bound: 554.9771138
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -123.2541962, 392.5810242, -125.6301498, 399.8784485, -523.1326294, 518.2111816
1: -173.5091553, 394.9635315, -176.8060608, 402.3511658, -575.8603516, 571.7695923
2: -146.6661682, 436.5891418, -149.4530487, 444.7727051, -591.4386597, 586.0421143
3: -154.5595398, 561.9185181, -157.5191956, 572.3540649, -726.9135742, 719.4376221
4: -131.8280182, 512.9211426, -134.3448944, 522.5493774, -654.3773804, 647.2659912

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9782949, upper bound: 554.9780110
time: 1.27 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9782949, upper bound: 554.9780110
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -137.6115265, 437.8572693, -125.6301498, 399.8784485, -537.4899902, 563.4874268
1: -193.6290741, 440.6151428, -176.8060608, 402.3511658, -595.9801636, 617.4212036
2: -163.5594940, 487.1738281, -149.4530487, 444.7727051, -608.3322144, 636.6268921
3: -172.4905701, 626.0422974, -157.5191956, 572.3540649, -744.8444214, 783.5614624
4: -146.9165649, 572.0592651, -134.3448944, 522.5493774, -669.4659424, 706.4041748

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9782949, upper bound: 554.9780110
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9782949, upper bound: 554.9780110
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -137.2063904, 436.5897827, -139.5428925, 443.6639099, -580.8703003, 576.1326904
1: -193.1229858, 439.2609863, -196.3027954, 446.4835205, -639.6065063, 635.5636597
2: -163.1250305, 485.6827393, -165.8249969, 493.6915588, -656.8165283, 651.5074463
3: -172.0321655, 624.1105347, -174.8911896, 634.3011475, -806.3333130, 799.0016479
4: -146.5214691, 570.3097534, -148.9574280, 579.7293091, -726.2507324, 719.2672119

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9722958, upper bound: 554.9740693
time: 1.38 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9722958, upper bound: 554.9740693
time: 1.49 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -165.0459595, 525.7300415, -131.1480408, 415.6406250, -580.6865234, 656.8780518
1: -232.7288208, 529.0870361, -184.3409882, 418.8788452, -651.6076050, 713.4280396
2: -196.1724548, 585.3873901, -155.6340942, 463.4671936, -659.6396484, 741.0214233
3: -207.6070404, 752.3815308, -164.3570404, 594.9967651, -802.6037598, 916.7384644
4: -176.5145264, 688.8870239, -139.9299774, 544.4902954, -721.0046387, 828.8170166

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9740693, upper bound: 554.9740693
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9740693, upper bound: 554.9740693
time: 1.19 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.99 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 0, lower bound: -554.9821432, upper bound: 554.9807618
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 0, lower bound: -554.9821432, upper bound: 554.9807618
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 0, lower bound: -554.9821432, upper bound: 554.9807618
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 0, lower bound: -554.9821432, upper bound: 554.9807618
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 0, lower bound: -554.9767678, upper bound: 554.9771653
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 0, lower bound: -554.9767678, upper bound: 554.9771653
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 0, lower bound: -554.9746480, upper bound: 554.9771138
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 0, lower bound: -554.9759107, upper bound: 554.9771138
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 0, lower bound: -554.9782949, upper bound: 554.9780110
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 0, lower bound: -554.9782949, upper bound: 554.9780110
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 0, lower bound: -554.9782949, upper bound: 554.9780110
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 0, lower bound: -554.9782949, upper bound: 554.9780110
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 0, lower bound: -554.9722958, upper bound: 554.9740693
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 0, lower bound: -554.9722958, upper bound: 554.9740693
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 0, lower bound: -554.9740693, upper bound: 554.9740693
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 0, lower bound: -554.9740693, upper bound: 554.9740693

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -115.3944626, 363.5161438, -115.3944626, 363.5161438, -478.9105835, 478.9105835
1: -161.7974091, 366.7948303, -161.7974091, 366.7948303, -528.5922241, 528.5922241
2: -136.8181305, 405.7499084, -136.8181305, 405.7499084, -542.5680542, 542.5680542
3: -144.2350464, 521.4130859, -144.2350464, 521.4130859, -665.6481323, 665.6481323
4: -123.1167984, 476.9207153, -123.1167984, 476.9207153, -600.0375366, 600.0375366

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9599447, upper bound: 554.9516137
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9599423, upper bound: 554.9539838
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -115.3944626, 363.5161438, -123.2541962, 392.5810242, -507.9754639, 486.7703247
1: -161.7974091, 366.7948303, -173.5091553, 394.9635315, -556.7608643, 540.3039551
2: -136.8181305, 405.7499084, -146.6661682, 436.5891418, -573.4072266, 552.4159546
3: -144.2350464, 521.4130859, -154.5595398, 561.9185181, -706.1535034, 675.9725952
4: -123.1167984, 476.9207153, -131.8280182, 512.9211426, -636.0379028, 608.7487183

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9599447, upper bound: 554.9516137
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9599423, upper bound: 554.9539838
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -131.5360107, 414.6815186, -115.3944626, 363.5161438, -495.0521545, 530.0759888
1: -184.4300995, 418.1284180, -161.7974091, 366.7948303, -551.2249146, 579.9256592
2: -155.8180084, 462.7027588, -136.8181305, 405.7499084, -561.5679321, 599.5208740
3: -164.3947754, 593.6577148, -144.2350464, 521.4130859, -685.8078613, 737.8927002
4: -140.0432434, 543.4032593, -123.1167984, 476.9207153, -616.9639893, 666.5200806

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9775086, upper bound: 554.9764934
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9821432, upper bound: 554.9807618
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -131.5360107, 414.6815186, -123.2541962, 392.5810242, -524.1170654, 537.9357300
1: -184.4300995, 418.1284180, -173.5091553, 394.9635315, -579.3936157, 591.6375732
2: -155.8180084, 462.7027588, -146.6661682, 436.5891418, -592.4070435, 609.3687744
3: -164.3947754, 593.6577148, -154.5595398, 561.9185181, -726.3132324, 748.2171631
4: -140.0432434, 543.4032593, -131.8280182, 512.9211426, -652.9643555, 675.2312622

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9775086, upper bound: 554.9764934
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9821432, upper bound: 554.9807618
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -130.4546814, 411.3610229, -130.8709869, 412.5370789, -542.9916992, 542.2319946
1: -182.9963989, 414.6822205, -183.5049438, 415.9659729, -598.9624023, 598.1871338
2: -154.5989075, 458.8901978, -155.0375366, 460.3154602, -614.9143677, 613.9276123
3: -163.1041565, 588.7929077, -163.5684509, 590.5667114, -753.6708984, 752.3613281
4: -138.9424438, 538.9245605, -139.3413696, 540.5904541, -679.5328369, 678.2659302

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9753373, upper bound: 554.9747466
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9767678, upper bound: 554.9771653
time: 1.32 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -130.4546814, 411.3610229, -136.9599457, 435.7474670, -566.2021484, 548.3209839
1: -182.9963989, 414.6822205, -192.7236328, 438.4920959, -621.4884644, 607.4057617
2: -154.5989075, 458.8901978, -162.7984467, 484.8243103, -639.4230957, 621.6885376
3: -163.1041565, 588.7929077, -171.6831360, 623.0082397, -786.1124268, 760.4760132
4: -138.9424438, 538.9245605, -146.2324066, 569.3012085, -708.2435913, 685.1569824

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9753373, upper bound: 554.9747466
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9767678, upper bound: 554.9771653
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -161.1584473, 510.1374817, -122.9428253, 386.0717468, -547.2301636, 633.0803223
1: -226.5817108, 514.0497437, -172.1863556, 389.9273376, -616.5089722, 686.2360840
2: -190.9772797, 568.9328613, -145.4021606, 431.6568298, -622.6340942, 714.3350220
3: -202.2574005, 730.7261353, -153.6018372, 553.6874390, -755.9448242, 884.3279419
4: -171.9734344, 669.8974609, -130.8190918, 507.3523865, -679.3258057, 800.7164917

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9678641, upper bound: 554.9696694
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9759107, upper bound: 554.9771138
time: 1.47 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -161.1584473, 510.1374817, -128.5524597, 407.7392273, -568.8976440, 638.6899414
1: -226.5817108, 514.0497437, -180.7563171, 410.8908691, -637.4724121, 694.8060303
2: -190.9772797, 568.9328613, -152.6003876, 454.5989990, -645.5762939, 721.5332642
3: -202.2574005, 730.7261353, -161.1440735, 583.7315063, -785.9888916, 891.8702393
4: -171.9734344, 669.8974609, -137.2028198, 534.0908813, -706.0643311, 807.1002197

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9678641, upper bound: 554.9696694
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9759107, upper bound: 554.9771138
time: 1.28 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -123.2541962, 392.5810242, -115.3944626, 363.5161438, -486.7703247, 507.9754639
1: -173.5091553, 394.9635315, -161.7974091, 366.7948303, -540.3039551, 556.7608643
2: -146.6661682, 436.5891418, -136.8181305, 405.7499084, -552.4159546, 573.4072266
3: -154.5595398, 561.9185181, -144.2350464, 521.4130859, -675.9725952, 706.1535034
4: -131.8280182, 512.9211426, -123.1167984, 476.9207153, -608.7487183, 636.0379028

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9577946, upper bound: 554.9501900
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9493292, upper bound: 554.9493292
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -123.2541962, 392.5810242, -123.2541962, 392.5810242, -515.8352051, 515.8352051
1: -173.5091553, 394.9635315, -173.5091553, 394.9635315, -568.4726562, 568.4726562
2: -146.6661682, 436.5891418, -146.6661682, 436.5891418, -583.2550659, 583.2550659
3: -154.5595398, 561.9185181, -154.5595398, 561.9185181, -716.4779663, 716.4779663
4: -131.8280182, 512.9211426, -131.8280182, 512.9211426, -644.7491455, 644.7491455

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9577946, upper bound: 554.9501900
time: 1.23 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9493292, upper bound: 554.9493292
time: 1.29 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -137.6115265, 437.8572693, -115.3944626, 363.5161438, -501.1276550, 553.2517090
1: -193.6290741, 440.6151428, -161.7974091, 366.7948303, -560.4238892, 602.4124146
2: -163.5594940, 487.1738281, -136.8181305, 405.7499084, -569.3093872, 623.9919434
3: -172.4905701, 626.0422974, -144.2350464, 521.4130859, -693.9035034, 770.2773438
4: -146.9165649, 572.0592651, -123.1167984, 476.9207153, -623.8372803, 695.1760864

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9418322, upper bound: 554.9613785
time: 1.35 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9783970, upper bound: 554.9762986
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -137.6115265, 437.8572693, -123.2541962, 392.5810242, -530.1925659, 561.1114502
1: -193.6290741, 440.6151428, -173.5091553, 394.9635315, -588.5925293, 614.1242676
2: -163.5594940, 487.1738281, -146.6661682, 436.5891418, -600.1486206, 633.8397827
3: -172.4905701, 626.0422974, -154.5595398, 561.9185181, -734.4088745, 780.6018066
4: -146.9165649, 572.0592651, -131.8280182, 512.9211426, -659.8377075, 703.8872681

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9418322, upper bound: 554.9653458
time: 1.10 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9783970, upper bound: 554.9762986
time: 1.51 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -137.2063904, 436.5897827, -130.8709869, 412.5370789, -549.7434692, 567.4607544
1: -193.1229858, 439.2609863, -183.5049438, 415.9659729, -609.0888672, 622.7658691
2: -163.1250305, 485.6827393, -155.0375366, 460.3154602, -623.4404907, 640.7200928
3: -172.0321655, 624.1105347, -163.5684509, 590.5667114, -762.5988159, 787.6789551
4: -146.5214691, 570.3097534, -139.3413696, 540.5904541, -687.1118774, 709.6511230

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9067646, upper bound: 554.9159782
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9722958, upper bound: 554.9740693
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -137.2063904, 436.5897827, -136.9599457, 435.7474670, -572.9538574, 573.5497437
1: -193.1229858, 439.2609863, -192.7236328, 438.4920959, -631.6149902, 631.9844360
2: -163.1250305, 485.6827393, -162.7984467, 484.8243103, -647.9490967, 648.4810181
3: -172.0321655, 624.1105347, -171.6831360, 623.0082397, -795.0403442, 795.7936401
4: -146.5214691, 570.3097534, -146.2324066, 569.3012085, -715.8226929, 716.5421753

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9067646, upper bound: 554.9159782
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9722958, upper bound: 554.9740693
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -165.0459595, 525.7300415, -122.9428253, 386.0717468, -551.1176758, 648.6728516
1: -232.7288208, 529.0870361, -172.1863556, 389.9273376, -622.6561279, 701.2733765
2: -196.1724548, 585.3873901, -145.4021606, 431.6568298, -627.8292847, 730.7895508
3: -207.6070404, 752.3815308, -153.6018372, 553.6874390, -761.2944336, 905.9833374
4: -176.5145264, 688.8870239, -130.8190918, 507.3523865, -683.8668213, 819.7060547

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9698118, upper bound: 554.9633016
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9621604, upper bound: 554.9621453
time: 1.23 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -165.0459595, 525.7300415, -128.5524597, 407.7392273, -572.7851562, 654.2824707
1: -232.7288208, 529.0870361, -180.7563171, 410.8908691, -643.6195679, 709.8433838
2: -196.1724548, 585.3873901, -152.6003876, 454.5989990, -650.7714844, 737.9877319
3: -207.6070404, 752.3815308, -161.1440735, 583.7315063, -791.3385620, 913.5255737
4: -176.5145264, 688.8870239, -137.2028198, 534.0908813, -710.6053467, 826.0897827

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9698118, upper bound: 554.9633016
time: 1.43 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9621604, upper bound: 554.9621453
time: 1.21 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 6.28 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9599447, upper bound: 554.9516137
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9599423, upper bound: 554.9539838
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9599447, upper bound: 554.9516137
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9599423, upper bound: 554.9539838
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9775086, upper bound: 554.9764934
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9821432, upper bound: 554.9807618
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9775086, upper bound: 554.9764934
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9821432, upper bound: 554.9807618
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9753373, upper bound: 554.9747466
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9767678, upper bound: 554.9771653
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9753373, upper bound: 554.9747466
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9767678, upper bound: 554.9771653
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9678641, upper bound: 554.9696694
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9759107, upper bound: 554.9771138
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9678641, upper bound: 554.9696694
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9759107, upper bound: 554.9771138
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9577946, upper bound: 554.9501900
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9493292, upper bound: 554.9493292
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9577946, upper bound: 554.9501900
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9493292, upper bound: 554.9493292
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9418322, upper bound: 554.9613785
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9783970, upper bound: 554.9762986
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9418322, upper bound: 554.9653458
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9783970, upper bound: 554.9762986
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9067646, upper bound: 554.9159782
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9722958, upper bound: 554.9740693
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9067646, upper bound: 554.9159782
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9722958, upper bound: 554.9740693
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9698118, upper bound: 554.9633016
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9621604, upper bound: 554.9621453
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9698118, upper bound: 554.9633016
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.28
Output dim: 0, lower bound: -554.9621604, upper bound: 554.9621453

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -123.8264008, 391.1094055, -113.5666351, 357.3257446, -481.1521606, 504.6760254
1: -173.4983215, 394.4565735, -159.2304382, 360.6926575, -534.1908569, 553.6870117
2: -146.6014099, 436.4610291, -134.6635895, 398.9460449, -545.5474854, 571.1245728
3: -154.7384491, 560.1813354, -141.9550934, 512.5579224, -667.2963867, 702.1364136
4: -131.8179932, 512.5811157, -121.2009506, 468.8694763, -600.6874390, 633.7820435

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9848150, upper bound: 554.9850678
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9863685, upper bound: 554.9859846
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -129.3510437, 407.1157227, -114.1490402, 359.1965637, -488.5476074, 521.2647705
1: -181.2432861, 410.7232971, -159.9902802, 362.5722046, -543.8154907, 570.7135620
2: -153.1453857, 454.5035400, -135.3014221, 401.0746155, -554.2199707, 589.8049316
3: -161.5780792, 583.0626831, -142.6346588, 515.3881836, -676.9661865, 725.6972656
4: -137.6862946, 533.7856445, -121.7742844, 471.4414062, -609.1276245, 655.5596924

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9844368, upper bound: 554.9844955
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9859959, upper bound: 554.9852815
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -123.8264008, 391.1094055, -121.6551514, 387.3326111, -511.1589966, 512.7645264
1: -173.4983215, 394.4565735, -171.2542267, 389.7282410, -563.2265625, 565.7107544
2: -146.6014099, 436.4610291, -144.7697601, 430.7543945, -577.3557739, 581.2307739
3: -154.7384491, 560.1813354, -152.5598145, 554.3616333, -709.1000977, 712.7410889
4: -131.8179932, 512.5811157, -130.1418152, 506.0134888, -637.8314819, 642.7229004

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.8439642, upper bound: 554.7946681
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9740002, upper bound: 554.9726646
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -129.3510437, 407.1157227, -121.9048843, 387.8716125, -517.2226562, 529.0206299
1: -181.2432861, 410.7232971, -171.5601654, 390.3339539, -571.5772095, 582.2834473
2: -153.1453857, 454.5035400, -145.0305328, 431.4729309, -584.6182861, 599.5340576
3: -161.5780792, 583.0626831, -152.8312988, 555.2838745, -716.8619385, 735.8939209
4: -137.6862946, 533.7856445, -130.3754730, 506.9208374, -644.6071167, 664.1610107

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9791652, upper bound: 554.9722729
time: 1.37 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9815517, upper bound: 554.9791358
time: 1.18 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -122.7936172, 388.0506897, -129.0525970, 406.3819580, -529.1755981, 517.1032715
1: -172.1464386, 391.2060242, -180.9530487, 409.8963013, -582.0426025, 572.1590576
2: -145.4258118, 432.8863831, -152.8986511, 453.5566711, -598.9824829, 585.7849731
3: -153.5052185, 555.5686646, -161.3008881, 581.7841797, -735.2893677, 716.8695679
4: -130.7545166, 508.3491211, -137.4380493, 532.6141357, -663.3685913, 645.7871704

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9769300, upper bound: 554.9761968
time: 1.22 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9725015, upper bound: 554.9738342
time: 1.52 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9693787, upper bound: 554.9662772
time: 1.26 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -128.2213440, 403.6400757, -129.6276550, 408.2326965, -536.4539795, 533.2677002
1: -179.7428284, 407.1215515, -181.6926270, 411.7539978, -591.4968262, 588.8141479
2: -151.8737183, 450.5242004, -153.5190125, 455.6502686, -607.5239258, 604.0432129
3: -160.2281494, 577.9760132, -161.9662476, 584.5393066, -744.7674561, 739.9422607
4: -136.5369263, 529.1081543, -138.0015564, 535.1176147, -671.6543579, 667.1096191

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9769387, upper bound: 554.9765995
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9732639, upper bound: 554.9748286
time: 1.29 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9693835, upper bound: 554.9663940
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -122.7936172, 388.0506897, -135.2385712, 430.0193176, -552.8129272, 523.2891846
1: -172.1464386, 391.2060242, -190.2944946, 432.8167419, -604.9631348, 581.5004883
2: -145.4258118, 432.8863831, -160.7617340, 478.5155029, -623.9412231, 593.6478882
3: -153.5052185, 555.5686646, -169.5285187, 614.8300781, -768.3352661, 725.0971680
4: -130.7545166, 508.3491211, -144.4168549, 561.8408203, -692.5952148, 652.7659912

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9280163, upper bound: 554.9035545
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9691694, upper bound: 554.9669775
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689510, upper bound: 554.9662449
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -128.2213440, 403.6400757, -135.6221161, 431.0928040, -559.3140259, 539.2620850
1: -179.7428284, 407.1215515, -190.7850952, 433.9213867, -613.6641846, 597.9066162
2: -151.8737183, 450.5242004, -161.1719971, 479.7678223, -631.6414795, 611.6961670
3: -160.2281494, 577.9760132, -169.9674835, 616.4545898, -776.6826782, 747.9434814
4: -136.5369263, 529.1081543, -144.7932892, 563.3666382, -699.9033813, 673.9014282

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9732135, upper bound: 554.9720579
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699635, upper bound: 554.9701909
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689801, upper bound: 554.9663752
time: 1.27 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -155.1871338, 492.8115845, -120.9771881, 379.3763428, -534.5634766, 613.7886963
1: -218.1627045, 496.4179688, -169.4129639, 383.3370361, -601.4996338, 665.8308105
2: -183.8057709, 549.4038086, -143.0800323, 424.3262939, -608.1320190, 692.4838257
3: -194.7823792, 706.0087891, -151.1359558, 544.1680908, -738.9504395, 857.1447144
4: -165.5705719, 646.9511108, -128.7505646, 498.7032166, -664.2738037, 775.7016602

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9590513, upper bound: 554.9659370
time: 1.28 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9572324, upper bound: 554.9604395
time: 1.64 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -158.7494507, 501.7432861, -121.7764130, 382.0158081, -540.7652588, 623.5197144
1: -223.0924377, 505.8561401, -170.4877319, 385.9534607, -609.0458984, 676.3438721
2: -188.0649567, 559.8839111, -143.9779205, 427.2570190, -615.3218384, 703.8618164
3: -199.1736145, 718.9594727, -152.1004333, 548.0236816, -747.1972656, 871.0599365
4: -169.3983002, 659.1768799, -129.5644073, 502.2004395, -671.5985718, 788.7412720

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9660108, upper bound: 554.9719835
time: 1.58 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9622561, upper bound: 554.9622561
time: 1.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -155.1871338, 492.8115845, -126.7624664, 401.7793579, -556.9664917, 619.5740356
1: -218.1627045, 496.4179688, -178.2227936, 404.9971008, -623.1597290, 674.6405029
2: -183.8057709, 549.4038086, -150.4790955, 448.0044861, -631.8102417, 699.8828735
3: -194.7823792, 706.0087891, -158.8951416, 575.2516479, -770.0339966, 864.9038696
4: -165.5705719, 646.9511108, -135.3097534, 526.3137207, -691.8842773, 782.2608032

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9539445, upper bound: 554.9583463
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9551131, upper bound: 554.9570640
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -158.7494507, 501.7432861, -127.2800751, 403.2493896, -561.9988403, 629.0233765
1: -223.0924377, 505.8561401, -178.9163208, 406.5164795, -629.6088867, 684.7724609
2: -188.0649567, 559.8839111, -151.0555573, 449.7900085, -637.8549194, 710.9394531
3: -199.1736145, 718.9594727, -159.5147705, 577.4369507, -776.6105347, 878.4742432
4: -169.3983002, 659.1768799, -135.8372192, 528.4580078, -697.8562012, 795.0140381

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9633016, upper bound: 554.9698119
time: 1.29 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9621453, upper bound: 554.9622650
time: 1.35 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -134.7304993, 428.3551331, -114.0278168, 359.0515442, -493.7820435, 542.3829346
1: -189.5167542, 431.1124573, -159.8442383, 362.3363037, -551.8530273, 590.9566650
2: -160.0974274, 476.7073669, -135.1689758, 400.8253174, -560.9227295, 611.8762207
3: -168.8433228, 612.4936523, -142.5016632, 515.0616455, -683.9049683, 754.9952393
4: -143.8236237, 559.7412109, -121.6418457, 471.1448364, -614.9683838, 681.3829956

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9753931, upper bound: 554.9761775
time: 1.32 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9811667, upper bound: 554.9812717
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9819859, upper bound: 554.9820001
time: 1.26 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -134.7304993, 428.3551331, -121.8874054, 388.0787659, -522.8092651, 550.2425537
1: -189.5167542, 431.1124573, -171.5510406, 390.4495544, -579.9662476, 602.6634521
2: -160.0974274, 476.7073669, -145.0172882, 431.6301575, -591.7276001, 621.7246704
3: -168.8433228, 612.4936523, -152.8246918, 555.4912109, -724.3345337, 765.3182983
4: -143.8236237, 559.7412109, -130.3538361, 507.0883179, -650.9119263, 690.0950317

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9370863, upper bound: 554.9240768
time: 1.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9783970, upper bound: 554.9762986
time: 1.42 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -134.6921539, 427.9428406, -129.6276550, 408.2326965, -542.9248047, 557.5704956
1: -189.4943390, 430.7486572, -181.6926270, 411.7539978, -601.2483521, 612.4412842
2: -160.0801544, 476.2491455, -153.5190125, 455.6502686, -615.7304077, 629.7680054
3: -168.8176727, 611.9271851, -161.9662476, 584.5393066, -753.3569946, 773.8933105
4: -143.8224945, 559.2651978, -138.0015564, 535.1176147, -678.9400635, 697.2667236

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=619.6494140625
rel_dist={0: [-554.990153595014, 554.9901535950139]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1102.43 seconds
