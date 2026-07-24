## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 5)
Time budget: 1800 seconds
Split limit: 100
Threshold: 27.5662213848


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0799713, 51.0799675)
1: (-19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778)
2: (-13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6210327, 29.6210327)
3: (-14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0640717, 37.0640640)
4: (-18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209)
5: (-16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765)
6: (-25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520)
7: (-23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790)
8: (-20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.4103546, 44.4103470)
9: (-14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724)
10: (-29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808)
11: (-33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669)
12: (-27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4790344, 39.4790382)
13: (-18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208)
14: (-56.6111145, -1.5055046, -56.6111145, -1.5055046, -50.0486603, 50.0486603)
15: (-21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448)
16: (-33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847)
17: (-62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.1339264, 62.1339340)
18: (-34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.9741211, 36.9741211)
19: (-27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667)
20: (-19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7679443, 28.7679482)
21: (-31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973)
22: (-32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4571991, 38.4571953)
23: (-23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009)
24: (-28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526)
25: (-22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5885162, 33.5885124)
26: (-34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.8358078, 43.8358040)
27: (-28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267)
28: (-22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785)
29: (-34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828)
30: (-25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035)
31: (-34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786)
32: (-20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219)
33: (-30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1827240, 51.1827164)
34: (-28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951)
35: (-25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409)
36: (-24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5372620, 43.5372620)
37: (-44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2858734, 58.2858810)
38: (-33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559)
39: (-34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3795395, 51.3795471)
40: (-34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.7035675, 49.7035675)
41: (-24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897)
42: (-16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.41 + 104.36 = 106.77 seconds
status: Status.UNKNOWN
relational distance
Output dim: 13, lower bound: -27.5938152, upper bound: 27.5938152

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 642

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 555

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5898428, upper bound: 27.5938114
time: 252.69 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5938113, upper bound: 27.5898428
time: 91.67 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 344.37 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 344.37
Output dim: 13, lower bound: -27.5898428, upper bound: 27.5938114
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 344.37
Output dim: 13, lower bound: -27.5938113, upper bound: 27.5898428

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0753860, 51.0761490
1: -19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778
2: -13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6181412, 29.6186790
3: -14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0529709, 37.0549622
4: -18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209
5: -16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765
6: -25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520
7: -23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790
8: -20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.4071045, 44.4079819
9: -14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724
10: -29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808
11: -33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669
12: -27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4628754, 39.4596405
13: -18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208
14: -56.6111145, -1.5055046, -56.6111145, -1.5055046, -50.0211792, 50.0137939
15: -21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448
16: -33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847
17: -62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.1174011, 62.1141968
18: -34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.9675179, 36.9632912
19: -27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667
20: -19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7606888, 28.7590256
21: -31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973
22: -32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4539490, 38.4527512
23: -23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009
24: -28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526
25: -22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5845985, 33.5837059
26: -34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.8160095, 43.8117065
27: -28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267
28: -22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785
29: -34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828
30: -25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035
31: -34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786
32: -20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219
33: -30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1700745, 51.1722488
34: -28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951
35: -25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409
36: -24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5382156, 43.5384369
37: -44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2784271, 58.2804413
38: -33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559
39: -34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3777847, 51.3779984
40: -34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.7001038, 49.7005768
41: -24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897
42: -16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 556

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 691

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5894337, upper bound: 27.5707851
time: 53.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5667923, upper bound: 27.5934026
time: 52.13 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0761490, 51.0753860
1: -19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778
2: -13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6186752, 29.6181412
3: -14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0549622, 37.0529671
4: -18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209
5: -16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765
6: -25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520
7: -23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790
8: -20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.4079895, 44.4070969
9: -14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724
10: -29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808
11: -33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669
12: -27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4596405, 39.4628754
13: -18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208
14: -56.6111145, -1.5055046, -56.6111145, -1.5055046, -50.0137939, 50.0211868
15: -21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448
16: -33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847
17: -62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.1141968, 62.1174011
18: -34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.9632912, 36.9675179
19: -27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667
20: -19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7590179, 28.7606888
21: -31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973
22: -32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4527512, 38.4539490
23: -23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009
24: -28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526
25: -22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5837135, 33.5845909
26: -34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.8117065, 43.8160019
27: -28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267
28: -22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785
29: -34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828
30: -25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035
31: -34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786
32: -20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219
33: -30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1722412, 51.1700668
34: -28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951
35: -25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409
36: -24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5384369, 43.5382156
37: -44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2804413, 58.2784348
38: -33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559
39: -34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3779984, 51.3777847
40: -34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.7005920, 49.7001114
41: -24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897
42: -16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1689

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 702

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5894053, upper bound: 27.5618178
time: 53.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5657869, upper bound: 27.5894053
time: 62.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 118.09 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 118.09
Output dim: 13, lower bound: -27.5894337, upper bound: 27.5707851
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 118.09
Output dim: 13, lower bound: -27.5667923, upper bound: 27.5934026
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 118.09
Output dim: 13, lower bound: -27.5894053, upper bound: 27.5618178
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 118.09
Output dim: 13, lower bound: -27.5657869, upper bound: 27.5894053

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0747833, 51.0752029
1: -19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778
2: -13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6146660, 29.6147499
3: -14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0527573, 37.0545845
4: -18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209
5: -16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765
6: -25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520
7: -23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790
8: -20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.4054413, 44.4061050
9: -14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724
10: -29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808
11: -33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669
12: -27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4634399, 39.4602928
13: -18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208
14: -56.6111145, -1.5055046, -56.6111145, -1.5055046, -50.0143890, 50.0072021
15: -21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448
16: -33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847
17: -62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.1174469, 62.1139450
18: -34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.9631157, 36.9593544
19: -27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667
20: -19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7573471, 28.7560577
21: -31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973
22: -32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4523392, 38.4513168
23: -23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009
24: -28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526
25: -22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5821457, 33.5815353
26: -34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.8151779, 43.8110695
27: -28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267
28: -22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785
29: -34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828
30: -25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035
31: -34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786
32: -20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219
33: -30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1699905, 51.1721268
34: -28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951
35: -25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409
36: -24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5398636, 43.5402832
37: -44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2813873, 58.2835846
38: -33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559
39: -34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3792572, 51.3796730
40: -34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.7023621, 49.7031555
41: -24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897
42: -16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 718

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 551

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5891754, upper bound: 27.5462408
time: 81.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5648956, upper bound: 27.5705277
time: 52.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0744476, 51.0755463
1: -19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778
2: -13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6142159, 29.6152000
3: -14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0525894, 37.0547523
4: -18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209
5: -16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765
6: -25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520
7: -23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790
8: -20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.4052277, 44.4063187
9: -14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724
10: -29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808
11: -33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669
12: -27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4635162, 39.4602165
13: -18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208
14: -56.6111145, -1.5055046, -56.6111145, -1.5055046, -50.0145874, 50.0069962
15: -21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448
16: -33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847
17: -62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.1171417, 62.1142426
18: -34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.9635811, 36.9588890
19: -27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667
20: -19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7577209, 28.7556839
21: -31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973
22: -32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4525146, 38.4511337
23: -23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009
24: -28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526
25: -22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5824203, 33.5812607
26: -34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.8153610, 43.8108864
27: -28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267
28: -22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785
29: -34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828
30: -25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035
31: -34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786
32: -20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219
33: -30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1699295, 51.1721725
34: -28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951
35: -25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409
36: -24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5400543, 43.5400848
37: -44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2815704, 58.2833939
38: -33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559
39: -34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3794556, 51.3794670
40: -34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.7026978, 49.7028275
41: -24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897
42: -16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 628

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 735

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5660732, upper bound: 27.5848737
time: 51.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5582295, upper bound: 27.5927108
time: 86.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0734253, 51.0722809
1: -19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778
2: -13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6150665, 29.6142273
3: -14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0522995, 37.0497818
4: -18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209
5: -16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765
6: -25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520
7: -23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790
8: -20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.4059448, 44.4046936
9: -14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724
10: -29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808
11: -33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669
12: -27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4567032, 39.4605255
13: -18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208
14: -56.6111145, -1.5055046, -56.6111145, -1.5055046, -50.0125275, 50.0201187
15: -21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448
16: -33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847
17: -62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.1109924, 62.1146545
18: -34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.9556885, 36.9611626
19: -27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667
20: -19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7546234, 28.7569504
21: -31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973
22: -32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4502106, 38.4518166
23: -23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009
24: -28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526
25: -22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5807953, 33.5820770
26: -34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.8063126, 43.8114929
27: -28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267
28: -22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785
29: -34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828
30: -25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035
31: -34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786
32: -20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219
33: -30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1726990, 51.1703415
34: -28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951
35: -25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409
36: -24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5371017, 43.5367050
37: -44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2791901, 58.2769012
38: -33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559
39: -34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3777008, 51.3773041
40: -34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.6985397, 49.6977234
41: -24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897
42: -16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 653

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 572

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5635306, upper bound: 27.5613590
time: 55.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5929152, upper bound: 27.5330123
time: 61.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0730438, 51.0726547
1: -19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778
2: -13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6147614, 29.6145363
3: -14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0517807, 37.0503082
4: -18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209
5: -16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765
6: -25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520
7: -23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790
8: -20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.4055786, 44.4050522
9: -14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724
10: -29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808
11: -33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669
12: -27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4572983, 39.4599342
13: -18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208
14: -56.6111145, -1.5055046, -56.6111145, -1.5055046, -50.0127258, 50.0199280
15: -21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448
16: -33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847
17: -62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.1114502, 62.1141968
18: -34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.9569321, 36.9599152
19: -27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667
20: -19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7552795, 28.7562904
21: -31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973
22: -32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4506149, 38.4514084
23: -23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009
24: -28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526
25: -22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5811920, 33.5816803
26: -34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.8071976, 43.8106079
27: -28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267
28: -22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785
29: -34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828
30: -25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035
31: -34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786
32: -20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219
33: -30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1725159, 51.1705246
34: -28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951
35: -25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409
36: -24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5369263, 43.5368805
37: -44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2789001, 58.2771835
38: -33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559
39: -34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3775177, 51.3774872
40: -34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.6982040, 49.6980667
41: -24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897
42: -16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 698

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 540

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5641007, upper bound: 27.5882106
time: 137.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5606231, upper bound: 27.5877234
time: 63.09 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 203.03 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 203.03
Output dim: 13, lower bound: -27.5891754, upper bound: 27.5462408
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 203.03
Output dim: 13, lower bound: -27.5648956, upper bound: 27.5705277
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 203.03
Output dim: 13, lower bound: -27.5660732, upper bound: 27.5848737
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 203.03
Output dim: 13, lower bound: -27.5582295, upper bound: 27.5927108
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 203.03
Output dim: 13, lower bound: -27.5635306, upper bound: 27.5613590
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 203.03
Output dim: 13, lower bound: -27.5929152, upper bound: 27.5330123
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 203.03
Output dim: 13, lower bound: -27.5641007, upper bound: 27.5882106
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 203.03
Output dim: 13, lower bound: -27.5606231, upper bound: 27.5877234

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0752182, 51.0756454
1: -19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778
2: -13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6141052, 29.6142502
3: -14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0518875, 37.0536156
4: -18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209
5: -16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765
6: -25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520
7: -23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790
8: -20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.4049911, 44.4056473
9: -14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724
10: -29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808
11: -33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669
12: -27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4629517, 39.4599838
13: -18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208
14: -56.6111145, -1.5055046, -56.6111145, -1.5055046, -50.0119171, 50.0051537
15: -21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448
16: -33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847
17: -62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.1170425, 62.1136932
18: -34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.9609756, 36.9574051
19: -27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667
20: -19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7574615, 28.7562294
21: -31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973
22: -32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4516983, 38.4507141
23: -23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009
24: -28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526
25: -22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5815926, 33.5809860
26: -34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.8135529, 43.8097000
27: -28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267
28: -22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785
29: -34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828
30: -25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035
31: -34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786
32: -20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219
33: -30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1704712, 51.1724777
34: -28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951
35: -25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409
36: -24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5398560, 43.5402603
37: -44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2817993, 58.2838974
38: -33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559
39: -34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3792877, 51.3796883
40: -34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.7023621, 49.7031479
41: -24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897
42: -16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 731

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 674

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5890723, upper bound: 27.5451234
time: 54.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5877107, upper bound: 27.5461251
time: 52.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0752182, 51.0756340
1: -19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778
2: -13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6141663, 29.6141891
3: -14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0517960, 37.0537109
4: -18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209
5: -16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765
6: -25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520
7: -23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790
8: -20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.4049606, 44.4056702
9: -14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724
10: -29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808
11: -33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669
12: -27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4631348, 39.4597931
13: -18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208
14: -56.6111145, -1.5055046, -56.6111145, -1.5055046, -50.0123444, 50.0047417
15: -21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448
16: -33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847
17: -62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.1171951, 62.1135406
18: -34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.9611588, 36.9572105
19: -27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667
20: -19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7575226, 28.7561684
21: -31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973
22: -32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4517365, 38.4506760
23: -23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009
24: -28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526
25: -22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5816002, 33.5809784
26: -34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.8138123, 43.8094406
27: -28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267
28: -22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785
29: -34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828
30: -25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035
31: -34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786
32: -20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219
33: -30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1703491, 51.1726074
34: -28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951
35: -25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409
36: -24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5398483, 43.5402679
37: -44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2817078, 58.2839890
38: -33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559
39: -34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3792572, 51.3796921
40: -34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.7023621, 49.7031555
41: -24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897
42: -16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 652

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 671

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5647566, upper bound: 27.5634552
time: 55.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5578378, upper bound: 27.5703785
time: 53.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0730438, 51.0739288
1: -19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778
2: -13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6124763, 29.6132011
3: -14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0516663, 37.0536804
4: -18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209
5: -16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765
6: -25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520
7: -23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790
8: -20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.4043579, 44.4052887
9: -14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724
10: -29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808
11: -33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669
12: -27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4626389, 39.4594574
13: -18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208
14: -56.6111145, -1.5055046, -56.6111145, -1.5055046, -50.0112457, 50.0037308
15: -21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448
16: -33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847
17: -62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.1144714, 62.1115494
18: -34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.9608154, 36.9565086
19: -27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667
20: -19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7558441, 28.7540932
21: -31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973
22: -32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4515762, 38.4503403
23: -23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009
24: -28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526
25: -22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5811310, 33.5801697
26: -34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.8138657, 43.8096008
27: -28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267
28: -22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785
29: -34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828
30: -25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035
31: -34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786
32: -20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219
33: -30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1698151, 51.1720276
34: -28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951
35: -25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409
36: -24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5391541, 43.5392990
37: -44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2799072, 58.2817459
38: -33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559
39: -34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3785095, 51.3786354
40: -34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.7011108, 49.7014236
41: -24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897
42: -16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 651

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 554

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5659556, upper bound: 27.5812687
time: 154.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5598482, upper bound: 27.5847044
time: 57.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0728149, 51.0741539
1: -19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778
2: -13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6122169, 29.6134682
3: -14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0515213, 37.0538330
4: -18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209
5: -16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765
6: -25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520
7: -23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790
8: -20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.4042053, 44.4054489
9: -14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724
10: -29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808
11: -33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669
12: -27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4627686, 39.4593353
13: -18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208
14: -56.6111145, -1.5055046, -56.6111145, -1.5055046, -50.0113220, 50.0036545
15: -21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448
16: -33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847
17: -62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.1144409, 62.1115723
18: -34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.9612045, 36.9561157
19: -27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667
20: -19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7561340, 28.7538071
21: -31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973
22: -32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4517212, 38.4501953
23: -23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009
24: -28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526
25: -22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5813293, 33.5799713
26: -34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.8140793, 43.8093910
27: -28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267
28: -22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785
29: -34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828
30: -25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035
31: -34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786
32: -20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219
33: -30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1697998, 51.1720352
34: -28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951
35: -25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409
36: -24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5392685, 43.5391846
37: -44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2799377, 58.2817307
38: -33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559
39: -34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3786316, 51.3785172
40: -34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.7012939, 49.7012253
41: -24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897
42: -16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 610

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 666

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5569978, upper bound: 27.5924865
time: 39.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5580375, upper bound: 27.5914659
time: 43.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0683212, 51.0657539
1: -19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778
2: -13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6090889, 29.6070557
3: -14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0365906, 37.0309677
4: -18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209
5: -16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765
6: -25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520
7: -23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790
8: -20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.3996811, 44.3969498
9: -14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724
10: -29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808
11: -33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669
12: -27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4275436, 39.4362144
13: -18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208
14: -56.6111145, -1.5055046, -56.6111145, -1.5055046, -49.9539032, 49.9716263
15: -21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448
16: -33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847
17: -62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.0980682, 62.1059189
18: -34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.9224930, 36.9346390
19: -27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667
20: -19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7377548, 28.7428703
21: -31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973
22: -32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4384613, 38.4420128
23: -23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009
24: -28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526
25: -22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5714264, 33.5742569
26: -34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.7672119, 43.7789001
27: -28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267
28: -22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785
29: -34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828
30: -25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035
31: -34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786
32: -20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219
33: -30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1635513, 51.1580429
34: -28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951
35: -25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409
36: -24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5370102, 43.5365067
37: -44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2854004, 58.2805023
38: -33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559
39: -34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3768539, 51.3763809
40: -34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.6976166, 49.6964722
41: -24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897
42: -16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 697

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1686

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5630890, upper bound: 27.5231418
time: 58.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5830440, upper bound: 27.5307726
time: 43.05 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 103.34 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 103.34
Output dim: 13, lower bound: -27.5890723, upper bound: 27.5451234
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 103.34
Output dim: 13, lower bound: -27.5877107, upper bound: 27.5461251
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 103.34
Output dim: 13, lower bound: -27.5647566, upper bound: 27.5634552
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 103.34
Output dim: 13, lower bound: -27.5578378, upper bound: 27.5703785
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 103.34
Output dim: 13, lower bound: -27.5659556, upper bound: 27.5812687
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 103.34
Output dim: 13, lower bound: -27.5598482, upper bound: 27.5847044
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 103.34
Output dim: 13, lower bound: -27.5569978, upper bound: 27.5924865
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 103.34
Output dim: 13, lower bound: -27.5580375, upper bound: 27.5914659
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 103.34
Output dim: 13, lower bound: -27.5630890, upper bound: 27.5231418
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 103.34
Output dim: 13, lower bound: -27.5830440, upper bound: 27.5307726
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 103.34
Output dim: 13, lower bound: -27.5641007, upper bound: 27.5882106
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 103.34
Output dim: 13, lower bound: -27.5606231, upper bound: 27.5877234

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 106.77 + 1790.99 = 1897.75 seconds
