## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 5)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.016598636200000003


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4100981, 0.4100981)
1: (-4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5186764, 0.5186763)
2: (-0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348959, 0.0348959)
3: (-0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1294576, 0.1294576)
4: (-0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1249279, 0.1249279)
5: (-0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1541328, 0.1541328)
6: (0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0408004, 0.0408004)
7: (-0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171396, 0.3171396)
8: (-5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3570098, 0.3570098)
9: (-4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3603758, 0.3603759)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.79 + 17.85 = 25.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0166300, upper bound: 0.0166398

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2419
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2841
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2826
type: A, layer: 1, pos: 3118
type: A, layer: 1, pos: 2827
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 3053
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 2839
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2336
type: A, layer: 1, pos: 2335
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2840
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 2334
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 2823
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 2259
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 2810
type: A, layer: 1, pos: 3293
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 2554
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2416
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 3026
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2919
type: A, layer: 1, pos: 2920
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3119

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2423

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0163828, upper bound: 0.0166293
time: 35.00 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166289, upper bound: 0.0166330
time: 16.21 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 51.27 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 51.27
Output dim: 4, lower bound: -0.0163828, upper bound: 0.0166293
NS_A2, status: Status.UNKNOWN, split count: 1, time: 51.27
Output dim: 4, lower bound: -0.0166289, upper bound: 0.0166330

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -4.4281168, -3.7377100, -4.4281168, -3.7373497, -0.4072609, 0.4068887
1: -4.9976158, -4.0740738, -4.9976163, -4.0732746, -0.5123008, 0.5114177
2: -0.5029410, -0.2839484, -0.5029488, -0.2839484, -0.0348263, 0.0348354
3: -0.5183766, -0.3154267, -0.5184115, -0.3154266, -0.1291532, 0.1291908
4: -0.2399203, 0.0868893, -0.2402469, 0.0868900, -0.1221730, 0.1225244
5: -0.9777650, -0.6966136, -0.9778252, -0.6966132, -0.1536596, 0.1537190
6: 0.3157849, 0.5166148, 0.3157787, 0.5166151, -0.0407354, 0.0407429
7: -0.9934496, -0.5435876, -0.9935354, -0.5435879, -0.3162968, 0.3163891
8: -5.7480974, -5.1392345, -5.7480984, -5.1385584, -0.3519439, 0.3513622
9: -4.4941730, -3.9372184, -4.4941750, -3.9366107, -0.3554109, 0.3546906

Time for backsubstitution: 5.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2841
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2826
type: B, layer: 1, pos: 3118
type: B, layer: 1, pos: 2827
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 2839
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 2336
type: B, layer: 1, pos: 2335
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 2853
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2840
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 2334
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 2823
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 2259
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 2810
type: B, layer: 1, pos: 3293
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 2554
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2416
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 3026
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3119

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3068

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0163538, upper bound: 0.0165384
time: 3.00 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0163543, upper bound: 0.0166005
time: 137.89 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -4.4325314, -3.7349899, -4.4281178, -3.7350171, -0.4147804, 0.4072037
1: -5.0071692, -4.0682707, -4.9976163, -4.0681744, -0.5295088, 0.5119073
2: -0.5029876, -0.2838715, -0.5029739, -0.2839484, -0.0348873, 0.0350104
3: -0.5189572, -0.3149350, -0.5186412, -0.3154261, -0.1294376, 0.1298113
4: -0.2425679, 0.0910196, -0.2425683, 0.0868943, -0.1223944, 0.1291708
5: -0.9781713, -0.6959248, -0.9781671, -0.6966132, -0.1537125, 0.1547175
6: 0.3157700, 0.5166981, 0.3157662, 0.5166166, -0.0407385, 0.0409377
7: -0.9940193, -0.5425823, -0.9940109, -0.5435876, -0.3168020, 0.3184438
8: -5.7564430, -5.1337976, -5.7481003, -5.1337957, -0.3655987, 0.3518212
9: -4.5017128, -3.9327159, -4.4941902, -3.9326415, -0.3691306, 0.3550744

Time for backsubstitution: 5.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2841
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2826
type: B, layer: 1, pos: 3118
type: B, layer: 1, pos: 2827
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 2839
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 2336
type: B, layer: 1, pos: 2335
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 2853
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2840
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 2334
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 2823
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 2259
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 2810
type: B, layer: 1, pos: 3293
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 2554
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2416
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 3026
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3119

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3068

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165995, upper bound: 0.0165442
time: 2.67 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165996, upper bound: 0.0165997
time: 77.40 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 86.06 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 86.06
Output dim: 4, lower bound: -0.0163538, upper bound: 0.0165384
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 86.06
Output dim: 4, lower bound: -0.0163543, upper bound: 0.0166005
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 86.06
Output dim: 4, lower bound: -0.0165995, upper bound: 0.0165442
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 86.06
Output dim: 4, lower bound: -0.0165996, upper bound: 0.0165997

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -4.4281158, -3.7378073, -4.4281158, -3.7374582, -0.3908593, 0.4067395
1: -4.9976139, -4.0741549, -4.9976139, -4.0733690, -0.4915439, 0.5112309
2: -0.5029176, -0.2839484, -0.5029215, -0.2839484, -0.0348133, 0.0347927
3: -0.5183356, -0.3154283, -0.5183643, -0.3154286, -0.1291018, 0.1249914
4: -0.2396984, 0.0868893, -0.2399871, 0.0868900, -0.1221251, 0.1185094
5: -0.9777442, -0.6966148, -0.9778005, -0.6966147, -0.1535960, 0.1486161
6: 0.3158118, 0.5166149, 0.3158103, 0.5166149, -0.0407163, 0.0385689
7: -0.9933321, -0.5435878, -0.9933983, -0.5435878, -0.3161364, 0.3160489
8: -5.7480974, -5.1392908, -5.7480974, -5.1386213, -0.3433813, 0.3511891
9: -4.4941726, -3.9372840, -4.4941740, -3.9366868, -0.3492512, 0.3546350

Time for backsubstitution: 5.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2419
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2841
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2826
type: A, layer: 1, pos: 3118
type: A, layer: 1, pos: 2827
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 3053
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2839
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2336
type: A, layer: 1, pos: 2335
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2840
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 2334
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 2823
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 2259
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 2810
type: A, layer: 1, pos: 3293
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 2554
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2416
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 3026
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2919
type: A, layer: 1, pos: 2920
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3119

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2420

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0161639, upper bound: 0.0165877
time: 17.57 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0163410, upper bound: 0.0165937
time: 42.42 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -4.4324789, -3.7457485, -4.4220338, -3.7474105, -0.4029166, 0.3911356
1: -5.0070724, -4.0814724, -4.9908094, -4.0831180, -0.5146838, 0.4918706
2: -0.5029551, -0.2838715, -0.5029290, -0.2838961, -0.0348394, 0.0349383
3: -0.5160056, -0.3150334, -0.5152498, -0.3171774, -0.1250401, 0.1265373
4: -0.2405738, 0.0910187, -0.2403132, 0.0865396, -0.1185863, 0.1263518
5: -0.9744468, -0.6960049, -0.9738851, -0.6987455, -0.1482670, 0.1506585
6: 0.3171357, 0.5166976, 0.3173217, 0.5159222, -0.0386668, 0.0394133
7: -0.9935498, -0.5425824, -0.9934835, -0.5435144, -0.3163711, 0.3178975
8: -5.7564430, -5.1389165, -5.7458935, -5.1395688, -0.3593917, 0.3431944
9: -4.5017071, -3.9366903, -4.4920268, -3.9371920, -0.3647468, 0.3491832

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2419
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2841
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2826
type: A, layer: 1, pos: 3118
type: A, layer: 1, pos: 2827
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 3053
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 2839
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2336
type: A, layer: 1, pos: 2335
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2840
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 2334
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 2823
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 2259
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 2810
type: A, layer: 1, pos: 3293
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 2554
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2416
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 3026
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2919
type: A, layer: 1, pos: 2920
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3119

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2420

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0164091, upper bound: 0.0165294
time: 2.53 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165864, upper bound: 0.0165272
time: 3.90 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -4.4325304, -3.7350874, -4.4281168, -3.7351258, -0.3983788, 0.4070544
1: -5.0071673, -4.0683508, -4.9976149, -4.0682688, -0.5087519, 0.5117201
2: -0.5029642, -0.2838715, -0.5029466, -0.2839484, -0.0348742, 0.0349678
3: -0.5189160, -0.3149367, -0.5185942, -0.3154282, -0.1293862, 0.1256120
4: -0.2423456, 0.0910195, -0.2423085, 0.0868942, -0.1223467, 0.1251559
5: -0.9781505, -0.6959262, -0.9781424, -0.6966149, -0.1536489, 0.1496146
6: 0.3157967, 0.5166980, 0.3157977, 0.5166169, -0.0407193, 0.0387638
7: -0.9939020, -0.5425823, -0.9938737, -0.5435874, -0.3166416, 0.3181032
8: -5.7564430, -5.1338539, -5.7480998, -5.1338587, -0.3570359, 0.3516476
9: -4.5017128, -3.9327807, -4.4941897, -3.9327188, -0.3629707, 0.3550186

Time for backsubstitution: 5.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2419
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2841
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2826
type: A, layer: 1, pos: 3118
type: A, layer: 1, pos: 2827
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 3053
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2839
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2336
type: A, layer: 1, pos: 2335
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2840
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 2334
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 2823
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 2259
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 2810
type: A, layer: 1, pos: 3293
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 2554
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2416
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 3026
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2919
type: A, layer: 1, pos: 2920
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3119

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2420

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0164112, upper bound: 0.0165941
time: 6.43 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165874, upper bound: 0.0165234
time: 73.23 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 85.68 seconds
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 85.68
Output dim: 4, lower bound: -0.0161639, upper bound: 0.0165877
NS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 85.68
Output dim: 4, lower bound: -0.0163410, upper bound: 0.0165937
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 85.68
Output dim: 4, lower bound: -0.0164091, upper bound: 0.0165294
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 85.68
Output dim: 4, lower bound: -0.0165864, upper bound: 0.0165272
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 85.68
Output dim: 4, lower bound: -0.0164112, upper bound: 0.0165941
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 85.68
Output dim: 4, lower bound: -0.0165874, upper bound: 0.0165234

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 25.63 + 448.41 = 474.05 seconds
