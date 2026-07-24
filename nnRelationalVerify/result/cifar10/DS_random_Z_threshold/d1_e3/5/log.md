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
execution time: IAR + RelationalAnalysis = 8.01 + 17.75 = 25.76 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0166300, upper bound: 0.0166398

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2513

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 762

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166294, upper bound: 0.0166325
time: 2.91 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166243, upper bound: 0.0166345
time: 2.47 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.40 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.40
Output dim: 4, lower bound: -0.0166294, upper bound: 0.0166325
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.40
Output dim: 4, lower bound: -0.0166243, upper bound: 0.0166345

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4096287, 0.4095937
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5182977, 0.5182681
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348687, 0.0348689
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1293375, 0.1293456
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1248809, 0.1248776
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1539830, 0.1539946
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407958, 0.0407958
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171167, 0.3171189
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3565140, 0.3564750
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3600964, 0.3600751

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2810

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 724

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166295, upper bound: 0.0166313
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166283, upper bound: 0.0166246
time: 96.30 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4095937, 0.4096287
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5182680, 0.5182977
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348689, 0.0348687
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1293456, 0.1293375
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1248776, 0.1248809
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1539946, 0.1539830
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407958, 0.0407958
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171189, 0.3171168
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3564749, 0.3565141
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3600751, 0.3600963

Time for backsubstitution: 6.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 692

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 778

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166260, upper bound: 0.0166297
time: 9.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166253, upper bound: 0.0166362
time: 4.08 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 19.90 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 19.90
Output dim: 4, lower bound: -0.0166295, upper bound: 0.0166313
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 19.90
Output dim: 4, lower bound: -0.0166283, upper bound: 0.0166246
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 19.90
Output dim: 4, lower bound: -0.0166260, upper bound: 0.0166297
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 19.90
Output dim: 4, lower bound: -0.0166253, upper bound: 0.0166362

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4095927, 0.4095556
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5180676, 0.5180173
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348590, 0.0348594
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1292761, 0.1292907
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1248677, 0.1248637
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1539140, 0.1539320
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407859, 0.0407869
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171144, 0.3171163
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3564628, 0.3564223
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3599536, 0.3599186

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3081

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165732, upper bound: 0.0165773
time: 20.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165805, upper bound: 0.0165707
time: 16.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4095906, 0.4095576
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5180471, 0.5180378
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348592, 0.0348591
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1292827, 0.1292841
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1248670, 0.1248644
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1539204, 0.1539256
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407869, 0.0407859
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171144, 0.3171163
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3564615, 0.3564237
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3599398, 0.3599322

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2470

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2418

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165486, upper bound: 0.0166102
time: 20.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166123, upper bound: 0.0165520
time: 4.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4093996, 0.4094046
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5180931, 0.5180933
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348603, 0.0348602
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1293036, 0.1292989
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1248719, 0.1248750
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1539381, 0.1539333
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407957, 0.0407957
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171015, 0.3170998
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3562545, 0.3562554
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3599229, 0.3599203

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2617

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3028

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166232, upper bound: 0.0166294
time: 21.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166210, upper bound: 0.0166337
time: 2.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4093695, 0.4094353
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5180635, 0.5181227
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348604, 0.0348601
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1293070, 0.1292956
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1248717, 0.1248752
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1539449, 0.1539265
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407957, 0.0407956
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171020, 0.3170996
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3562163, 0.3562933
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3598989, 0.3599441

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2418

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3022

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165905, upper bound: 0.0166328
time: 2.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166171, upper bound: 0.0165990
time: 2.70 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 11.59 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 11.59
Output dim: 4, lower bound: -0.0165732, upper bound: 0.0165773
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 11.59
Output dim: 4, lower bound: -0.0165805, upper bound: 0.0165707
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 11.59
Output dim: 4, lower bound: -0.0165486, upper bound: 0.0166102
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 11.59
Output dim: 4, lower bound: -0.0166123, upper bound: 0.0165520
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 11.59
Output dim: 4, lower bound: -0.0166232, upper bound: 0.0166294
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 11.59
Output dim: 4, lower bound: -0.0166210, upper bound: 0.0166337
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 11.59
Output dim: 4, lower bound: -0.0165905, upper bound: 0.0166328
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 11.59
Output dim: 4, lower bound: -0.0166171, upper bound: 0.0165990

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4093317, 0.4092710
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5169468, 0.5167997
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348573, 0.0348561
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1292728, 0.1292711
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1245798, 0.1245973
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1539180, 0.1539217
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407869, 0.0407859
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3170996, 0.3171016
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3551473, 0.3548545
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3586192, 0.3584049

Time for backsubstitution: 6.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 713

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2827

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165132, upper bound: 0.0166034
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165377, upper bound: 0.0165736
time: 56.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4093040, 0.4092987
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5168090, 0.5169375
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348562, 0.0348572
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1292698, 0.1292742
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1245999, 0.1245772
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1539166, 0.1539232
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407869, 0.0407859
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3170995, 0.3171016
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3548922, 0.3551096
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3584123, 0.3586117

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3022

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2617

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165589, upper bound: 0.0164766
time: 17.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165411, upper bound: 0.0164949
time: 117.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4083124, 0.4082339
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5171418, 0.5170827
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348056, 0.0348113
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1290865, 0.1291139
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1248294, 0.1248311
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1536328, 0.1536720
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407857, 0.0407866
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3170109, 0.3170198
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3549437, 0.3548559
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3591139, 0.3590755

Time for backsubstitution: 6.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 3078

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2826

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165879, upper bound: 0.0166144
time: 2.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166067, upper bound: 0.0165937
time: 35.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4082291, 0.4083161
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5170822, 0.5171416
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348113, 0.0348055
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1291184, 0.1290818
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1248279, 0.1248322
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1536762, 0.1536280
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407866, 0.0407857
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3170213, 0.3170092
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3548550, 0.3549443
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3590783, 0.3591101

Time for backsubstitution: 6.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 775

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 691

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166188, upper bound: 0.0166265
time: 64.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166167, upper bound: 0.0166287
time: 15.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.3950546, 0.3946685
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.4917753, 0.4911323
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0344033, 0.0344116
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1255866, 0.1256502
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1239116, 0.1239415
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1496857, 0.1497182
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0361187, 0.0362650
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3170757, 0.3170805
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3516759, 0.3516881
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3530312, 0.3528841

Time for backsubstitution: 6.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 3127

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 752

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165905, upper bound: 0.0166121
time: 99.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165816, upper bound: 0.0166286
time: 42.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.3946028, 0.3951203
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.4910733, 0.4918344
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0344118, 0.0344031
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1256616, 0.1255752
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1239380, 0.1239151
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1497366, 0.1496673
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0362651, 0.0361187
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3170829, 0.3170733
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3516111, 0.3517530
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3528388, 0.3530763

Time for backsubstitution: 6.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 689

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3065

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165600, upper bound: 0.0165874
time: 40.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166034, upper bound: 0.0165429
time: 35.88 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 82.34 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 82.34
Output dim: 4, lower bound: -0.0165132, upper bound: 0.0166034
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 82.34
Output dim: 4, lower bound: -0.0165377, upper bound: 0.0165736
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 82.34
Output dim: 4, lower bound: -0.0165589, upper bound: 0.0164766
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 82.34
Output dim: 4, lower bound: -0.0165411, upper bound: 0.0164949
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 82.34
Output dim: 4, lower bound: -0.0165879, upper bound: 0.0166144
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 82.34
Output dim: 4, lower bound: -0.0166067, upper bound: 0.0165937
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 82.34
Output dim: 4, lower bound: -0.0166188, upper bound: 0.0166265
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 82.34
Output dim: 4, lower bound: -0.0166167, upper bound: 0.0166287
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 82.34
Output dim: 4, lower bound: -0.0165905, upper bound: 0.0166121
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 82.34
Output dim: 4, lower bound: -0.0165816, upper bound: 0.0166286
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 82.34
Output dim: 4, lower bound: -0.0165600, upper bound: 0.0165874
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 82.34
Output dim: 4, lower bound: -0.0166034, upper bound: 0.0165429

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4063777, 0.4062819
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5122159, 0.5120270
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0345896, 0.0345811
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1283761, 0.1283825
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1234834, 0.1235161
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1529420, 0.1529398
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0401723, 0.0401872
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3170954, 0.3170962
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3538990, 0.3536059
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3571367, 0.3569084

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 704

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3028

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165105, upper bound: 0.0165999
time: 12.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165083, upper bound: 0.0166027
time: 51.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4049802, 0.4048601
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5119194, 0.5118120
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0345432, 0.0345424
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1280897, 0.1281292
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1236516, 0.1236684
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1525384, 0.1525807
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0401271, 0.0401443
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3170006, 0.3170094
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3534743, 0.3533781
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3574682, 0.3574097

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 732

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 708

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165852, upper bound: 0.0166109
time: 54.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165851, upper bound: 0.0166095
time: 8.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4049385, 0.4049019
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5118712, 0.5118599
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0345367, 0.0345489
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1281019, 0.1281170
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1236667, 0.1236533
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1525415, 0.1525776
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0401434, 0.0401279
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3170006, 0.3170094
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3534657, 0.3533866
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3574480, 0.3574297

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2825

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165876, upper bound: 0.0165885
time: 95.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166025, upper bound: 0.0165778
time: 36.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4080588, 0.4081347
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5168856, 0.5169510
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0347043, 0.0347025
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1291176, 0.1290810
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1247841, 0.1247858
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1536767, 0.1536285
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407753, 0.0407747
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3170162, 0.3170041
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3544651, 0.3545406
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3590577, 0.3590889

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 723

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2421

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0164101, upper bound: 0.0166051
time: 8.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165989, upper bound: 0.0164216
time: 2.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4080475, 0.4081459
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5168915, 0.5169450
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0347083, 0.0346985
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1291175, 0.1290810
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1247815, 0.1247884
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1536767, 0.1536285
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407757, 0.0407744
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3170161, 0.3170042
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3544514, 0.3545544
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3590571, 0.3590896

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 706

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2554

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166153, upper bound: 0.0166294
time: 16.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166152, upper bound: 0.0166273
time: 23.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.3947144, 0.3942680
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.4913476, 0.4906316
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0343408, 0.0343497
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1254553, 0.1255385
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1238938, 0.1239177
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1495351, 0.1495915
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0361123, 0.0362587
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3170385, 0.3170490
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3513284, 0.3512881
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3527277, 0.3525311

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 755

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2554

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165892, upper bound: 0.0166183
time: 15.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165891, upper bound: 0.0166163
time: 2.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.3946540, 0.3943284
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.4912747, 0.4907045
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0343415, 0.0343491
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1254748, 0.1255190
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1238879, 0.1239236
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1495590, 0.1495677
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0361124, 0.0362587
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3170443, 0.3170433
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3512759, 0.3513405
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3526782, 0.3525805

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2475

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3092

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165319, upper bound: 0.0166225
time: 3.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165788, upper bound: 0.0165826
time: 16.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.3942403, 0.3949847
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.4905629, 0.4916632
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0344051, 0.0343988
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1255950, 0.1254095
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1238847, 0.1238225
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1496584, 0.1494689
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0362625, 0.0361137
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3170800, 0.3170522
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3514701, 0.3517163
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3526303, 0.3530025

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3093

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2661

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165862, upper bound: 0.0164333
time: 56.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0164937, upper bound: 0.0165237
time: 6.99 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 69.79 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 69.79
Output dim: 4, lower bound: -0.0165105, upper bound: 0.0165999
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 69.79
Output dim: 4, lower bound: -0.0165083, upper bound: 0.0166027
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 69.79
Output dim: 4, lower bound: -0.0165852, upper bound: 0.0166109
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 69.79
Output dim: 4, lower bound: -0.0165851, upper bound: 0.0166095
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 69.79
Output dim: 4, lower bound: -0.0165876, upper bound: 0.0165885
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 69.79
Output dim: 4, lower bound: -0.0166025, upper bound: 0.0165778
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 69.79
Output dim: 4, lower bound: -0.0164101, upper bound: 0.0166051
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 69.79
Output dim: 4, lower bound: -0.0165989, upper bound: 0.0164216
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 69.79
Output dim: 4, lower bound: -0.0166153, upper bound: 0.0166294
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 69.79
Output dim: 4, lower bound: -0.0166152, upper bound: 0.0166273
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 69.79
Output dim: 4, lower bound: -0.0165892, upper bound: 0.0166183
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 69.79
Output dim: 4, lower bound: -0.0165891, upper bound: 0.0166163
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 69.79
Output dim: 4, lower bound: -0.0165319, upper bound: 0.0166225
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 69.79
Output dim: 4, lower bound: -0.0165788, upper bound: 0.0165826
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 69.79
Output dim: 4, lower bound: -0.0165862, upper bound: 0.0164333
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 69.79
Output dim: 4, lower bound: -0.0164937, upper bound: 0.0165237

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4052895, 0.4051117
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5112646, 0.5110166
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0345349, 0.0345322
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1281586, 0.1281969
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1234405, 0.1234722
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1526362, 0.1526774
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0401623, 0.0401781
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3170047, 0.3170158
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3525861, 0.3522047
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3563279, 0.3560649

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 732

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 766

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165113, upper bound: 0.0165900
time: 32.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165032, upper bound: 0.0165993
time: 7.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4052074, 0.4051934
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5112054, 0.5110757
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0345406, 0.0345265
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1281905, 0.1281649
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1234394, 0.1234733
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1526795, 0.1526340
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0401632, 0.0401772
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3170149, 0.3170053
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3524978, 0.3522933
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3562933, 0.3560996

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 3090

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 691

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165075, upper bound: 0.0166029
time: 2.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165054, upper bound: 0.0166010
time: 41.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4049118, 0.4047834
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5118879, 0.5117800
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0344573, 0.0344599
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1280709, 0.1281127
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1236272, 0.1236412
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1525160, 0.1525609
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0400994, 0.0401176
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3169895, 0.3169984
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3531306, 0.3530217
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3572799, 0.3572150

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 775

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 718

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165851, upper bound: 0.0166140
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165851, upper bound: 0.0166113
time: 2.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4049037, 0.4047916
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5118874, 0.5117805
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0344607, 0.0344565
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1280731, 0.1281106
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1236244, 0.1236441
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1525187, 0.1525583
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0401003, 0.0401166
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3169894, 0.3169985
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3531181, 0.3530343
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3572736, 0.3572214

Time for backsubstitution: 6.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 723

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 695

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165846, upper bound: 0.0166094
time: 17.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165849, upper bound: 0.0166120
time: 70.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4041239, 0.4041415
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5105530, 0.5106418
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0344607, 0.0344797
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1278701, 0.1278668
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1234892, 0.1234643
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1522867, 0.1523122
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0400636, 0.0400420
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3169780, 0.3169863
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3530724, 0.3530121
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3569595, 0.3569800

Time for backsubstitution: 6.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2839

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3091

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165950, upper bound: 0.0165755
time: 22.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166020, upper bound: 0.0165699
time: 33.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4081084, 0.4079738
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5169208, 0.5165981
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0347038, 0.0347024
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1291037, 0.1290654
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1245930, 0.1248054
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1536554, 0.1536075
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407725, 0.0407717
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3170066, 0.3169992
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3545025, 0.3542156
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3590475, 0.3587365

Time for backsubstitution: 6.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2646

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3093

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0163667, upper bound: 0.0165625
time: 32.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0164090, upper bound: 0.0165681
time: 2.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4078980, 0.4081347
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5165329, 0.5169510
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0347041, 0.0347025
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1291176, 0.1290671
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1247841, 0.1245948
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1536767, 0.1536072
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407753, 0.0407720
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3170162, 0.3169943
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3541402, 0.3545406
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3587055, 0.3590889

Time for backsubstitution: 6.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2826

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 761

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165975, upper bound: 0.0164075
time: 137.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165922, upper bound: 0.0164186
time: 5.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4075459, 0.4078388
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5159348, 0.5163299
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0346868, 0.0346755
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1289838, 0.1288856
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1247713, 0.1247786
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1535159, 0.1533940
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407443, 0.0407387
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3169928, 0.3169661
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3541446, 0.3543333
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3585985, 0.3587620

Time for backsubstitution: 6.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 3068

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2260

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166117, upper bound: 0.0166329
time: 2.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166157, upper bound: 0.0166247
time: 27.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4077403, 0.4076443
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5162765, 0.5159882
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0346852, 0.0346771
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1289222, 0.1289471
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1247717, 0.1247782
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1534423, 0.1534677
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407399, 0.0407430
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3169780, 0.3169808
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3542303, 0.3542476
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3587295, 0.3586311

Time for backsubstitution: 6.55 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 25.76 + 1776.14 = 1801.90 seconds
