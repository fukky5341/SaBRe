## Execution arguments:
Dataset: Dataset.CIFAR10
Network: onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 6)
Time budget: 7200 seconds
Split limit: 100
Threshold: 0.32578885785


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4069926, 1.4069926)
1: (-3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9955665, 1.9955665)
2: (-1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378)
3: (-1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8541119, 0.8541120)
4: (-0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7479159, 0.7479158)
5: (-1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7824583, 0.7824583)
6: (-0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4135126, 0.4135125)
7: (-2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8645149, 0.8645149)
8: (-3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1990520, 1.1990521)
9: (-3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0880301, 1.0880303)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 5.05 + 102.22 = 107.27 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.3268303, upper bound: 0.3268329

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 3538
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 3597

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 2119

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3266355, upper bound: 0.3261069
time: 18.73 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3261003, upper bound: 0.3266418
time: 45.08 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 63.93 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 63.93
Output dim: 0, lower bound: -0.3266355, upper bound: 0.3261069
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 63.93
Output dim: 0, lower bound: -0.3261003, upper bound: 0.3266418

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4027319, 1.4025705
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9907607, 1.9905623
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8517675, 0.8517919
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7468389, 0.7468948
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7797693, 0.7797958
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4134561, 0.4134578
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8645007, 0.8645003
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1972950, 1.1972761
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0858300, 1.0857276

Time for backsubstitution: 4.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3538
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 3597

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 3538

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3265742, upper bound: 0.3260432
time: 101.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3265742, upper bound: 0.3260427
time: 115.94 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4025707, 1.4027317
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9905626, 1.9907607
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8517917, 0.8517676
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7468948, 0.7468390
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7797958, 0.7797693
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4134578, 0.4134561
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8645003, 0.8645007
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1972761, 1.1972950
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0857276, 1.0858300

Time for backsubstitution: 4.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3538
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 3597

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 3538

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3260381, upper bound: 0.3265791
time: 167.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3260381, upper bound: 0.3265822
time: 188.67 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 360.27 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 360.27
Output dim: 0, lower bound: -0.3265742, upper bound: 0.3260432
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 360.27
Output dim: 0, lower bound: -0.3265742, upper bound: 0.3260427
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 360.27
Output dim: 0, lower bound: -0.3260381, upper bound: 0.3265791
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 360.27
Output dim: 0, lower bound: -0.3260381, upper bound: 0.3265822

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4028155, 1.4025700
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9907593, 1.9905680
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8517637, 0.8517811
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7467820, 0.7469018
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7797646, 0.7797678
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4134561, 0.4134576
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8644600, 0.8645222
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1972845, 1.1972780
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0858293, 1.0857278

Time for backsubstitution: 3.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 3597

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 598

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3265322, upper bound: 0.3260092
time: 21.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3265329, upper bound: 0.3260016
time: 105.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4027314, 1.4025705
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9907607, 1.9905609
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8517569, 0.8517919
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7468389, 0.7468379
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7797413, 0.7797958
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4134558, 0.4134578
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8645007, 0.8644596
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1972950, 1.1972656
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0858300, 1.0857269

Time for backsubstitution: 4.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 3597

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 598

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3265285, upper bound: 0.3260049
time: 487.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3265329, upper bound: 0.3260028
time: 90.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4026543, 1.4027312
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9905611, 1.9907659
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8517879, 0.8517569
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7468379, 0.7468460
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7797911, 0.7797413
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4134579, 0.4134558
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8644596, 0.8645225
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1972656, 1.1972969
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0857270, 1.0858303

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 3597

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 598

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3259943, upper bound: 0.3265430
time: 19.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3260025, upper bound: 0.3265404
time: 72.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4025702, 1.4027317
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9905626, 1.9907593
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8517811, 0.8517676
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7468948, 0.7467821
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7797678, 0.7797693
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4134576, 0.4134561
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8645003, 0.8644599
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1972761, 1.1972845
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0857276, 1.0858293

Time for backsubstitution: 4.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 3597

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 598

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3259943, upper bound: 0.3265456
time: 16.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3260025, upper bound: 0.3265382
time: 394.04 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 414.74 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 414.74
Output dim: 0, lower bound: -0.3265322, upper bound: 0.3260092
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 414.74
Output dim: 0, lower bound: -0.3265329, upper bound: 0.3260016
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 414.74
Output dim: 0, lower bound: -0.3265285, upper bound: 0.3260049
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 414.74
Output dim: 0, lower bound: -0.3265329, upper bound: 0.3260028
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 414.74
Output dim: 0, lower bound: -0.3259943, upper bound: 0.3265430
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 414.74
Output dim: 0, lower bound: -0.3260025, upper bound: 0.3265404
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 414.74
Output dim: 0, lower bound: -0.3259943, upper bound: 0.3265456
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 414.74
Output dim: 0, lower bound: -0.3260025, upper bound: 0.3265382

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4028044, 1.4025594
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9907585, 1.9905722
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8517580, 0.8517732
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7467786, 0.7468990
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7797577, 0.7797592
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4134561, 0.4134572
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8644570, 0.8645205
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1972716, 1.1972697
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0858259, 1.0857284

Time for backsubstitution: 4.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 3597

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 2585

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3256210, upper bound: 0.3257856
time: 31.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3263131, upper bound: 0.3251191
time: 411.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4028032, 1.4025590
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9907633, 1.9905674
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8517557, 0.8517743
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7467790, 0.7468983
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7797559, 0.7797598
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4134557, 0.4134575
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8644584, 0.8645192
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1972761, 1.1972651
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0858299, 1.0857246

Time for backsubstitution: 4.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 3597

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 2585

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3256203, upper bound: 0.3257816
time: 198.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3263170, upper bound: 0.3251154
time: 21.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4027203, 1.4025599
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9907597, 1.9905651
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8517501, 0.8517839
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7468355, 0.7468348
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7797334, 0.7797871
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4134558, 0.4134574
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8644978, 0.8644580
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1972821, 1.1972573
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0858266, 1.0857275

Time for backsubstitution: 4.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 3597

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 2585

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3256210, upper bound: 0.3257852
time: 178.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3263131, upper bound: 0.3251212
time: 720.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4027207, 1.4025595
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9907645, 1.9905603
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8517489, 0.8517850
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7468359, 0.7468344
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7797327, 0.7797878
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4134554, 0.4134578
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8644992, 0.8644566
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1972864, 1.1972529
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0858306, 1.0857234

Time for backsubstitution: 4.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 3597

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 2585

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3256203, upper bound: 0.3257856
time: 48.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3263170, upper bound: 0.3251180
time: 21.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4026432, 1.4027206
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9905604, 1.9907701
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8517823, 0.8517489
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7468344, 0.7468432
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7797841, 0.7797327
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4134579, 0.4134554
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8644566, 0.8645210
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1972528, 1.1972885
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0857234, 1.0858309

Time for backsubstitution: 4.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 3597

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 2585

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3251142, upper bound: 0.3263185
time: 75.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3257775, upper bound: 0.3256254
time: 186.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4026421, 1.4027202
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9905653, 1.9907653
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8517801, 0.8517501
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7468348, 0.7468425
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7797824, 0.7797334
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4134575, 0.4134558
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8644580, 0.8645195
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1972573, 1.1972840
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0857275, 1.0858269

Time for backsubstitution: 4.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 3597

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 2585

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3251146, upper bound: 0.3263192
time: 60.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3257770, upper bound: 0.3256233
time: 41.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4025592, 1.4027210
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9905617, 1.9907634
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8517743, 0.8517596
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7468913, 0.7467790
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7797598, 0.7797607
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4134576, 0.4134556
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8644974, 0.8644584
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1972632, 1.1972761
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0857241, 1.0858300

Time for backsubstitution: 4.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 3597

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 2585

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3251142, upper bound: 0.3263185
time: 813.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3257775, upper bound: 0.3256268
time: 199.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4025595, 1.4027207
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9905665, 1.9907587
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8517733, 0.8517607
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7468917, 0.7467786
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7797591, 0.7797613
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4134572, 0.4134561
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8644987, 0.8644570
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1972675, 1.1972717
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0857282, 1.0858259

Time for backsubstitution: 4.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 3597

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 2585

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3251146, upper bound: 0.3263171
time: 75.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3257770, upper bound: 0.3256215
time: 313.17 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 393.03 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 393.03
Output dim: 0, lower bound: -0.3256210, upper bound: 0.3257856
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 393.03
Output dim: 0, lower bound: -0.3263131, upper bound: 0.3251191
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 393.03
Output dim: 0, lower bound: -0.3256203, upper bound: 0.3257816
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 393.03
Output dim: 0, lower bound: -0.3263170, upper bound: 0.3251154
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 393.03
Output dim: 0, lower bound: -0.3256210, upper bound: 0.3257852
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 393.03
Output dim: 0, lower bound: -0.3263131, upper bound: 0.3251212
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 393.03
Output dim: 0, lower bound: -0.3256203, upper bound: 0.3257856
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 393.03
Output dim: 0, lower bound: -0.3263170, upper bound: 0.3251180
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 393.03
Output dim: 0, lower bound: -0.3251142, upper bound: 0.3263185
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 393.03
Output dim: 0, lower bound: -0.3257775, upper bound: 0.3256254
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 393.03
Output dim: 0, lower bound: -0.3251146, upper bound: 0.3263192
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 393.03
Output dim: 0, lower bound: -0.3257770, upper bound: 0.3256233
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 393.03
Output dim: 0, lower bound: -0.3251142, upper bound: 0.3263185
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 393.03
Output dim: 0, lower bound: -0.3257775, upper bound: 0.3256268
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 393.03
Output dim: 0, lower bound: -0.3251146, upper bound: 0.3263171
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 393.03
Output dim: 0, lower bound: -0.3257770, upper bound: 0.3256215

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4008877, 1.4004259
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9887804, 1.9883174
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8507050, 0.8507965
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7462792, 0.7464406
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7785347, 0.7786053
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4134504, 0.4134522
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8644050, 0.8644684
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1963484, 1.1962185
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0847569, 1.0845102

Time for backsubstitution: 4.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 3597

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 2120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3261620, upper bound: 0.3244684
time: 54.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3255155, upper bound: 0.3244642
time: 499.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4008867, 1.4004257
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9887851, 1.9883121
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8507028, 0.8507976
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7462796, 0.7464399
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7785329, 0.7786059
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4134500, 0.4134526
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8644065, 0.8644671
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1963530, 1.1962141
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0847609, 1.0845063

Time for backsubstitution: 4.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 3597

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 2120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3261657, upper bound: 0.3244684
time: 23.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3255171, upper bound: 0.3249656
time: 218.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4008037, 1.4004264
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9887816, 1.9883102
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8506970, 0.8508072
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7463361, 0.7463764
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7785102, 0.7786332
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4134502, 0.4134524
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8644458, 0.8644059
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1963589, 1.1962062
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0847576, 1.0845091

Time for backsubstitution: 4.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 3597

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 2120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3261617, upper bound: 0.3244675
time: 192.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3255148, upper bound: 0.3249691
time: 327.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4008040, 1.4004261
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9887863, 1.9883054
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8506960, 0.8508083
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7463365, 0.7463760
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7785096, 0.7786340
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4134497, 0.4134529
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8644471, 0.8644045
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1963632, 1.1962018
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0847615, 1.0845052

Time for backsubstitution: 4.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 3597

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 2120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3261657, upper bound: 0.3244647
time: 23.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3255171, upper bound: 0.3249661
time: 539.66 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 567.72 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 567.72
Output dim: 0, lower bound: -0.3261620, upper bound: 0.3244684
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 567.72
Output dim: 0, lower bound: -0.3255155, upper bound: 0.3244642
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 567.72
Output dim: 0, lower bound: -0.3261657, upper bound: 0.3244684
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 567.72
Output dim: 0, lower bound: -0.3255171, upper bound: 0.3249656
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 567.72
Output dim: 0, lower bound: -0.3261617, upper bound: 0.3244675
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 567.72
Output dim: 0, lower bound: -0.3255148, upper bound: 0.3249691
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 567.72
Output dim: 0, lower bound: -0.3261657, upper bound: 0.3244647
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 567.72
Output dim: 0, lower bound: -0.3255171, upper bound: 0.3249661
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 567.72
Output dim: 0, lower bound: -0.3251142, upper bound: 0.3263185
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 567.72
Output dim: 0, lower bound: -0.3251146, upper bound: 0.3263192
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 567.72
Output dim: 0, lower bound: -0.3251142, upper bound: 0.3263185
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 567.72
Output dim: 0, lower bound: -0.3251146, upper bound: 0.3263171

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 107.27 + 7203.24 = 7310.51 seconds
