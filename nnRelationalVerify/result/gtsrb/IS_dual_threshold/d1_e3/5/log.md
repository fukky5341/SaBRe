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
execution time: IAR + RelationalAnalysis = 2.38 + 106.25 = 108.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 13, lower bound: -27.5938152, upper bound: 27.5938152

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 645
type: B, layer: 1, pos: 645
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 729
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 605

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5428236, upper bound: 27.5854763
time: 39.38 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5428236, upper bound: 27.5925058
time: 57.45 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 97.00 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 97.00
Output dim: 13, lower bound: -27.5428236, upper bound: 27.5854763
IS_A2, status: Status.UNKNOWN, split count: 1, time: 97.00
Output dim: 13, lower bound: -27.5428236, upper bound: 27.5925058

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -36.9724350, 14.1625175, -36.9905167, 14.1754704, -51.0138855, 51.0217743
1: -19.7439575, 16.4470978, -19.7563934, 16.4545822, -36.1985397, 36.2034912
2: -13.5746861, 16.5940285, -13.6082287, 16.6117554, -29.5481720, 29.5643005
3: -13.9017372, 23.4508362, -13.9556408, 23.4844971, -36.9500122, 36.9702950
4: -18.6024933, 18.1306305, -18.6371002, 18.1527348, -36.7552261, 36.7677307
5: -16.0666924, 19.9922409, -16.1197453, 20.0225430, -36.0892334, 36.1119843
6: -25.9582539, 13.9980726, -25.9792843, 14.0053883, -39.9636421, 39.9773560
7: -23.3123703, 18.8709106, -23.3387489, 18.8858948, -42.1982651, 42.2096596
8: -20.6557007, 23.7192726, -20.6922455, 23.7433090, -44.3271027, 44.3421173
9: -14.7358189, 19.4574890, -14.7539091, 19.4753036, -34.2111206, 34.2113991
10: -29.7207317, 17.1267281, -29.7449112, 17.1747608, -46.8954926, 46.8716393
11: -33.7711716, 7.3890004, -33.7969551, 7.4370952, -41.2082672, 41.1859550
12: -27.9264946, 11.8673573, -27.9515915, 11.9116783, -39.4093246, 39.3815536
13: -18.0067749, 28.4605522, -18.0854950, 28.4915199, -46.4982948, 46.5460472
14: -56.5649986, -1.6014709, -56.5992432, -1.5506096, -49.9564819, 49.9373550
15: -21.7800102, 17.5726776, -21.8011703, 17.5853138, -39.3653259, 39.3738480
16: -33.0484505, 13.7200222, -33.0680275, 13.7556705, -46.8041229, 46.7880478
17: -62.9028549, 0.6584644, -62.9091492, 0.6800709, -62.1042633, 62.0838318
18: -34.8255424, 3.6379547, -34.8481941, 3.6956158, -36.8860855, 36.8498917
19: -27.2919197, 3.1198144, -27.3179741, 3.1529441, -30.4448643, 30.4377880
20: -19.1839790, 10.1539431, -19.1946507, 10.1817770, -28.7206955, 28.7065392
21: -31.7404861, 4.3496981, -31.7675209, 4.3833323, -36.1238174, 36.1172180
22: -32.1753311, 6.5296679, -32.1993561, 6.5630732, -38.3919525, 38.3824768
23: -23.3968925, 7.4482446, -23.4253254, 7.4973197, -30.8942127, 30.8735695
24: -28.0525131, 9.3722496, -28.0837555, 9.4237490, -37.4762611, 37.4560051
25: -21.9731503, 11.5699997, -21.9965343, 11.6094131, -33.5180244, 33.5022202
26: -34.8875275, 10.6712027, -34.9052544, 10.7291384, -43.7541656, 43.7171097
27: -28.7392025, 7.4830551, -28.7726192, 7.5368109, -36.2760124, 36.2556763
28: -22.4487114, 12.5685158, -22.4704990, 12.6102886, -35.0589981, 35.0390167
29: -34.3910942, 3.8877792, -34.4194946, 3.9217920, -38.3128853, 38.3072739
30: -25.8789825, 12.1681156, -25.8998795, 12.2043924, -38.0833740, 38.0679932
31: -34.2247543, 6.5468388, -34.2623901, 6.5926580, -40.8174133, 40.8092270
32: -20.6740246, 13.4111490, -20.6980648, 13.4333124, -34.1073380, 34.1092148
33: -30.0593948, 21.1905804, -30.1137161, 21.1844559, -51.0740814, 51.1332855
34: -28.8091469, 17.1063499, -28.8259888, 17.1355591, -45.9447060, 45.9323387
35: -25.8854485, 20.3002567, -25.9204292, 20.2970924, -46.1825409, 46.2206879
36: -24.5467625, 18.9636269, -24.5715485, 18.9759350, -43.4791794, 43.4916611
37: -44.6685524, 13.7372284, -44.7037888, 13.7683392, -58.1840973, 58.1979523
38: -33.0350151, 18.2825985, -33.0628967, 18.3149109, -51.3499260, 51.3454971
39: -34.6137238, 16.7903290, -34.6661911, 16.8074379, -51.2655182, 51.3011131
40: -34.5751228, 15.5468378, -34.5994415, 15.5623398, -49.6434174, 49.6522446
41: -24.5259972, 14.6367130, -24.5453529, 14.6616392, -39.1876373, 39.1820679
42: -16.4630890, 11.0691118, -16.4739246, 11.0862560, -27.5493450, 27.5430374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=119, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 645
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 645
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 573

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5376437, upper bound: 27.5520580
time: 57.11 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5376437, upper bound: 27.5849351
time: 109.50 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -37.0105858, 14.1829233, -37.0116959, 14.1869125, -51.0855713, 51.0751305
1: -19.7664852, 16.4576302, -19.7671356, 16.4596653, -36.2261505, 36.2247658
2: -13.6364088, 16.6171665, -13.6373224, 16.6176224, -29.6078033, 29.6199303
3: -14.0020943, 23.4920349, -14.0035601, 23.4927521, -37.0384064, 37.0622978
4: -18.6655102, 18.1607761, -18.6664238, 18.1614609, -36.8269730, 36.8272018
5: -16.1653442, 20.0286350, -16.1667824, 20.0292053, -36.1945496, 36.1954193
6: -25.9945183, 14.0101881, -25.9958191, 14.0112667, -40.0057831, 40.0060081
7: -23.3613338, 18.8897839, -23.3623619, 18.8903809, -42.2517166, 42.2521439
8: -20.7232780, 23.7528458, -20.7242870, 23.7538414, -44.3965759, 44.4083481
9: -14.7709475, 19.4890385, -14.7715979, 19.4902534, -34.2612000, 34.2606354
10: -29.7543888, 17.2151871, -29.7549858, 17.2166176, -46.9710083, 46.9701729
11: -33.8048744, 7.4773593, -33.8058167, 7.4785151, -41.2833900, 41.2831764
12: -27.9597721, 11.9486036, -27.9606915, 11.9498043, -39.4770355, 39.4479218
13: -18.1542969, 28.4979324, -18.1565018, 28.4986839, -46.6529808, 46.6544342
14: -56.6098022, -1.5072899, -56.6106949, -1.5060673, -50.0464935, 49.9839325
15: -21.8201561, 17.5942268, -21.8213348, 17.5949478, -39.4151039, 39.4155617
16: -33.0891914, 13.7892685, -33.0903435, 13.7905607, -46.8797531, 46.8796120
17: -62.9171829, 0.6933937, -62.9183273, 0.6960869, -62.1290436, 62.1574554
18: -34.8523445, 3.7466412, -34.8530121, 3.7481165, -36.9723282, 36.9208298
19: -27.3256950, 3.1825070, -27.3264694, 3.1833816, -30.5090771, 30.5089760
20: -19.1994591, 10.2050562, -19.2000866, 10.2058420, -28.7634926, 28.7443581
21: -31.7767296, 4.4131508, -31.7776604, 4.4140358, -36.1907654, 36.1908112
22: -32.2070351, 6.5923781, -32.2080154, 6.5932775, -38.4552917, 38.4403458
23: -23.4322891, 7.5397472, -23.4329491, 7.5409355, -30.9732246, 30.9726963
24: -28.0906601, 9.4689274, -28.0914726, 9.4702435, -37.5609055, 37.5604019
25: -22.0041046, 11.6442394, -22.0047455, 11.6453028, -33.5870743, 33.5737228
26: -34.9102364, 10.7778015, -34.9112091, 10.7792645, -43.8303299, 43.7856941
27: -28.7804623, 7.5843029, -28.7814331, 7.5855894, -36.3660507, 36.3657379
28: -22.4764614, 12.6464043, -22.4770603, 12.6474409, -35.1239014, 35.1234665
29: -34.4283638, 3.9513245, -34.4295883, 3.9521866, -38.3805504, 38.3809128
30: -25.9069519, 12.2351789, -25.9080524, 12.2361412, -38.1430931, 38.1432304
31: -34.2727547, 6.6338768, -34.2736740, 6.6351223, -40.9078751, 40.9073410
32: -20.7098808, 13.4536095, -20.7106590, 13.4543972, -34.1642761, 34.1642685
33: -30.1593227, 21.1922512, -30.1613159, 21.1927681, -51.1842422, 51.1808472
34: -28.8346634, 17.1609097, -28.8352509, 17.1618290, -45.9964905, 45.9961624
35: -25.9482536, 20.3023605, -25.9506836, 20.3027363, -46.2509918, 46.2530441
36: -24.5888958, 18.9859238, -24.5911980, 18.9867878, -43.5336304, 43.5350113
37: -44.7265472, 13.7947159, -44.7278214, 13.7955818, -58.2987061, 58.2823715
38: -33.0843201, 18.3418465, -33.0861320, 18.3428726, -51.4271927, 51.4279785
39: -34.7040253, 16.8232746, -34.7075768, 16.8239555, -51.3743210, 51.3769836
40: -34.6153183, 15.5761156, -34.6175308, 15.5767326, -49.7004852, 49.7016449
41: -24.5595856, 14.6840019, -24.5605850, 14.6851063, -39.2446899, 39.2445869
42: -16.4841156, 11.0993462, -16.4849663, 11.1005669, -27.5846825, 27.5843124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=119, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 645
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 573

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5376437, upper bound: 27.5603007
time: 58.69 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5919834, upper bound: 27.5919836
time: 75.14 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 136.07 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 136.07
Output dim: 13, lower bound: -27.5376437, upper bound: 27.5520580
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 136.07
Output dim: 13, lower bound: -27.5376437, upper bound: 27.5849351
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 136.07
Output dim: 13, lower bound: -27.5376437, upper bound: 27.5603007
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 136.07
Output dim: 13, lower bound: -27.5919834, upper bound: 27.5919836

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -36.9718513, 14.1600513, -36.9893303, 14.1701717, -51.0040283, 51.0195389
1: -19.7437859, 16.4459305, -19.7559795, 16.4520855, -36.1958694, 36.2019119
2: -13.5731678, 16.5937080, -13.6057959, 16.6110649, -29.5453339, 29.5569763
3: -13.8981009, 23.4504280, -13.9488039, 23.4836006, -36.9445801, 36.9690781
4: -18.5989075, 18.1301975, -18.6294632, 18.1518650, -36.7507706, 36.7596588
5: -16.0650940, 19.9918556, -16.1163864, 20.0216827, -36.0867767, 36.1082420
6: -25.9562130, 13.9977627, -25.9752159, 14.0047340, -39.9609451, 39.9729767
7: -23.3112488, 18.8705750, -23.3363457, 18.8852100, -42.1964569, 42.2069206
8: -20.6535969, 23.7187347, -20.6880798, 23.7422256, -44.3230438, 44.3452072
9: -14.7352810, 19.4562607, -14.7527752, 19.4730930, -34.2083740, 34.2090378
10: -29.7202759, 17.1253338, -29.7439632, 17.1718540, -46.8921280, 46.8692970
11: -33.7707405, 7.3881111, -33.7960625, 7.4352317, -41.2059708, 41.1841736
12: -27.9260311, 11.8662395, -27.9506874, 11.9093151, -39.3941345, 39.3790588
13: -18.0053444, 28.4601784, -18.0825043, 28.4906387, -46.4959831, 46.5426826
14: -56.5646057, -1.6030006, -56.5984802, -1.5538368, -49.8816528, 49.9348373
15: -21.7797012, 17.5713558, -21.8005219, 17.5824509, -39.3621521, 39.3718796
16: -33.0477180, 13.7192001, -33.0665207, 13.7538824, -46.8016014, 46.7857208
17: -62.9025688, 0.6573315, -62.9085388, 0.6776237, -62.0719299, 62.0818100
18: -34.8252220, 3.6367903, -34.8475075, 3.6931648, -36.8406677, 36.8463364
19: -27.2913971, 3.1188865, -27.3168812, 3.1509862, -30.4423828, 30.4357681
20: -19.1836700, 10.1525698, -19.1939926, 10.1788502, -28.7150497, 28.6998024
21: -31.7399883, 4.3492594, -31.7664738, 4.3823900, -36.1223793, 36.1157341
22: -32.1748886, 6.5289154, -32.1984482, 6.5614791, -38.3768387, 38.3808060
23: -23.3964157, 7.4465971, -23.4243279, 7.4939256, -30.8903408, 30.8709259
24: -28.0519352, 9.3705196, -28.0825424, 9.4200478, -37.4719849, 37.4530640
25: -21.9726067, 11.5686598, -21.9953918, 11.6065903, -33.5068817, 33.4990158
26: -34.8870316, 10.6699400, -34.9042282, 10.7264423, -43.7089081, 43.7148094
27: -28.7387791, 7.4821849, -28.7717667, 7.5349760, -36.2737541, 36.2539520
28: -22.4483223, 12.5669804, -22.4696598, 12.6069870, -35.0553093, 35.0366402
29: -34.3906555, 3.8871336, -34.4186172, 3.9204483, -38.3111038, 38.3057518
30: -25.8784561, 12.1675110, -25.8987694, 12.2031965, -38.0816536, 38.0662804
31: -34.2241631, 6.5455856, -34.2611694, 6.5902300, -40.8143921, 40.8067551
32: -20.6731148, 13.4101734, -20.6961823, 13.4312172, -34.1043320, 34.1063538
33: -30.0573215, 21.1903629, -30.1093636, 21.1839695, -51.0714417, 51.1070480
34: -28.8075123, 17.1060181, -28.8225880, 17.1348877, -45.9423981, 45.9286041
35: -25.8836784, 20.3000946, -25.9166832, 20.2967300, -46.1804085, 46.2167778
36: -24.5454960, 18.9634781, -24.5688629, 18.9756317, -43.4775467, 43.4869614
37: -44.6669960, 13.7370548, -44.7005386, 13.7680044, -58.1816711, 58.1778870
38: -33.0336266, 18.2822342, -33.0600014, 18.3141327, -51.3477592, 51.3422356
39: -34.6112976, 16.7901077, -34.6610565, 16.8070297, -51.2625732, 51.2938385
40: -34.5738525, 15.5466862, -34.5969086, 15.5620289, -49.6416855, 49.6458130
41: -24.5247841, 14.6365204, -24.5428238, 14.6612129, -39.1859970, 39.1793442
42: -16.4625835, 11.0650110, -16.4728699, 11.0785294, -27.5411129, 27.5378799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=118, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 645
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 637

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5402804, upper bound: 27.5556689
time: 55.79 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5356107, upper bound: 27.5510195
time: 46.39 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -37.0100365, 14.1804562, -37.0105362, 14.1816235, -51.0757523, 51.0728760
1: -19.7663059, 16.4564476, -19.7667141, 16.4571571, -36.2234650, 36.2231598
2: -13.6349325, 16.6168461, -13.6349163, 16.6169319, -29.6049461, 29.6126022
3: -13.9984617, 23.4915886, -13.9967346, 23.4918232, -37.0329895, 37.0611954
4: -18.6618919, 18.1603661, -18.6587486, 18.1606064, -36.8224983, 36.8191147
5: -16.1637611, 20.0282269, -16.1633949, 20.0283585, -36.1921196, 36.1916199
6: -25.9925117, 14.0098820, -25.9917603, 14.0106220, -40.0031357, 40.0016403
7: -23.3602142, 18.8894272, -23.3599548, 18.8896751, -42.2498894, 42.2493820
8: -20.7212048, 23.7523212, -20.7201710, 23.7527580, -44.3924713, 44.4114304
9: -14.7704105, 19.4878197, -14.7704544, 19.4880638, -34.2584763, 34.2582741
10: -29.7539291, 17.2138519, -29.7540207, 17.2136974, -46.9676285, 46.9678726
11: -33.8044434, 7.4764447, -33.8049049, 7.4766378, -41.2810822, 41.2813492
12: -27.9593220, 11.9474859, -27.9597588, 11.9474478, -39.4622650, 39.4454498
13: -18.1528778, 28.4975185, -18.1535034, 28.4978104, -46.6506882, 46.6510239
14: -56.6094246, -1.5088310, -56.6099014, -1.5092945, -49.9716644, 49.9813919
15: -21.8198395, 17.5928841, -21.8206825, 17.5921211, -39.4119606, 39.4135666
16: -33.0884857, 13.7884064, -33.0888443, 13.7887659, -46.8772507, 46.8772507
17: -62.9168549, 0.6922264, -62.9176598, 0.6936150, -62.0967484, 62.1554871
18: -34.8520279, 3.7454786, -34.8523102, 3.7456484, -36.9269447, 36.9172668
19: -27.3251762, 3.1816034, -27.3253937, 3.1814198, -30.5065956, 30.5069962
20: -19.1991348, 10.2036839, -19.1994362, 10.2029171, -28.7578812, 28.7376671
21: -31.7762108, 4.4126925, -31.7766075, 4.4130850, -36.1892967, 36.1893005
22: -32.2066002, 6.5916104, -32.2071114, 6.5917091, -38.4401932, 38.4386978
23: -23.4318085, 7.5380931, -23.4319611, 7.5375471, -30.9693565, 30.9700546
24: -28.0900841, 9.4671917, -28.0902462, 9.4665432, -37.5566254, 37.5574379
25: -22.0035515, 11.6428986, -22.0036030, 11.6424942, -33.5759659, 33.5704880
26: -34.9097137, 10.7765036, -34.9101486, 10.7765999, -43.7851257, 43.7833824
27: -28.7800598, 7.5834427, -28.7805595, 7.5837460, -36.3638077, 36.3640022
28: -22.4760609, 12.6448402, -22.4762535, 12.6441383, -35.1202011, 35.1210938
29: -34.4279747, 3.9506941, -34.4286804, 3.9508257, -38.3787994, 38.3793755
30: -25.9064274, 12.2345762, -25.9069386, 12.2349072, -38.1413345, 38.1415138
31: -34.2721634, 6.6326199, -34.2724533, 6.6327181, -40.9048805, 40.9031258
32: -20.7089748, 13.4526291, -20.7087669, 13.4522839, -34.1612587, 34.1613960
33: -30.1572723, 21.1920166, -30.1569672, 21.1922588, -51.1815796, 51.1546021
34: -28.8330154, 17.1605778, -28.8319054, 17.1611557, -45.9941711, 45.9924850
35: -25.9464645, 20.3022194, -25.9469490, 20.3023777, -46.2488403, 46.2491684
36: -24.5876141, 18.9857674, -24.5885201, 18.9864655, -43.5319672, 43.5303268
37: -44.7249832, 13.7945528, -44.7245178, 13.7952347, -58.2962799, 58.2623062
38: -33.0829582, 18.3414803, -33.0832901, 18.3421173, -51.4250755, 51.4247704
39: -34.7015800, 16.8230896, -34.7024117, 16.8235283, -51.3713989, 51.3697395
40: -34.6140671, 15.5759640, -34.6150513, 15.5764236, -49.6988068, 49.6952209
41: -24.5583839, 14.6837931, -24.5580463, 14.6846800, -39.2430649, 39.2418404
42: -16.4835968, 11.0952387, -16.4839134, 11.0928097, -27.5764065, 27.5791512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=118, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 645
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 645
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 637

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5905480, upper bound: 27.5631011
time: 45.32 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5908503, upper bound: 27.5908502
time: 64.58 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 112.11 seconds
IS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 112.11
Output dim: 13, lower bound: -27.5402804, upper bound: 27.5556689
IS_A1_B2_B2, status: Status.VERIFIED, split count: 3, time: 112.11
Output dim: 13, lower bound: -27.5356107, upper bound: 27.5510195
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 112.11
Output dim: 13, lower bound: -27.5905480, upper bound: 27.5631011
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 112.11
Output dim: 13, lower bound: -27.5908503, upper bound: 27.5908502

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -36.9859772, 14.1755762, -36.9462700, 14.1612816, -51.0277252, 51.0022621
1: -19.7349186, 16.4524612, -19.6999817, 16.4137268, -36.1486435, 36.1524429
2: -13.6035442, 16.6125793, -13.5691624, 16.5713291, -29.5271072, 29.5422401
3: -13.9482555, 23.4854279, -13.8922787, 23.4138088, -36.9041672, 36.9505806
4: -18.6237335, 18.1539841, -18.5783348, 18.0936108, -36.7173462, 36.7323189
5: -16.1149712, 20.0227890, -16.0629616, 19.9542255, -36.0691986, 36.0857506
6: -25.9809456, 14.0037136, -25.9536572, 14.0035191, -39.9844666, 39.9573708
7: -23.3194313, 18.8864346, -23.2749939, 18.8276939, -42.1471252, 42.1614304
8: -20.6816788, 23.7478161, -20.6364460, 23.6855850, -44.2854004, 44.3229523
9: -14.7438564, 19.4831734, -14.7106905, 19.4593830, -34.2032394, 34.1938629
10: -29.7401981, 17.2045059, -29.7256489, 17.1854095, -46.9256058, 46.9301529
11: -33.7924652, 7.4528322, -33.7435913, 7.4244485, -41.2169151, 41.1964226
12: -27.9534893, 11.9143000, -27.9053497, 11.8737698, -39.3807144, 39.3535080
13: -18.1130333, 28.4919891, -18.0680847, 28.4429436, -46.5559769, 46.5600739
14: -56.5961113, -1.5369968, -56.5461693, -1.5691376, -49.8940277, 49.8915520
15: -21.7942295, 17.5855560, -21.7640572, 17.5468273, -39.3410568, 39.3496132
16: -33.0660629, 13.7830639, -33.0315742, 13.7698488, -46.8359108, 46.8146362
17: -62.9001312, 0.6761398, -62.8679123, 0.6510773, -62.0371780, 62.0925140
18: -34.8463669, 3.7049685, -34.8017998, 3.6601305, -36.8336601, 36.8242455
19: -27.3179302, 3.1481862, -27.2617073, 3.1137943, -30.4317245, 30.4098930
20: -19.1928921, 10.1822176, -19.1734123, 10.1563730, -28.7046509, 28.6879082
21: -31.7659378, 4.3862863, -31.7193336, 4.3597064, -36.1256447, 36.1056213
22: -32.1992950, 6.5625720, -32.1506958, 6.5304661, -38.3715363, 38.3526688
23: -23.4246368, 7.4926138, -23.3592949, 7.4445829, -30.8692207, 30.8519096
24: -28.0841656, 9.4226913, -28.0213032, 9.3741970, -37.4583626, 37.4439926
25: -21.9969559, 11.6044159, -21.9449291, 11.5627594, -33.4896202, 33.4731636
26: -34.9037285, 10.7311840, -34.8586044, 10.6809740, -43.6830902, 43.6856117
27: -28.7741966, 7.5526376, -28.7296448, 7.5199203, -36.2941170, 36.2822838
28: -22.4702854, 12.6048985, -22.4161682, 12.5609818, -35.0312653, 35.0210648
29: -34.4186401, 3.9252920, -34.3656044, 3.8966417, -38.3152809, 38.2908974
30: -25.8986149, 12.2081013, -25.8588257, 12.1772690, -38.0758820, 38.0669250
31: -34.2637177, 6.5897198, -34.1938438, 6.5438766, -40.8075943, 40.7815399
32: -20.7007980, 13.4364910, -20.6642799, 13.4175768, -34.1183739, 34.1007690
33: -30.1451912, 21.1691437, -30.1092358, 21.1439743, -51.1163330, 51.0681763
34: -28.8266087, 17.1251202, -28.7804108, 17.0882244, -45.9148331, 45.9055328
35: -25.9385300, 20.2784328, -25.8990860, 20.2541771, -46.1927071, 46.1775208
36: -24.5813980, 18.9626675, -24.5442142, 18.9386520, -43.4778214, 43.4622498
37: -44.7081833, 13.7461109, -44.6238289, 13.6963596, -58.1772766, 58.1001892
38: -33.0752029, 18.3155136, -33.0367088, 18.2842999, -51.3595047, 51.3522224
39: -34.6851425, 16.7945042, -34.6278152, 16.7649918, -51.2961426, 51.2656174
40: -34.6000443, 15.5587273, -34.5613289, 15.5397930, -49.6478882, 49.6226501
41: -24.5472450, 14.6614685, -24.5021324, 14.6376038, -39.1848488, 39.1636009
42: -16.4733658, 11.0876617, -16.4461517, 11.0733528, -27.5467186, 27.5338135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=117, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 645
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 683

## Relational analysis of IS_A2_B2_B1_B1

### Relational analysis result of IS_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5216913, upper bound: 27.5611163
time: 47.15 seconds

## Relational analysis of IS_A2_B2_B1_B2

### Relational analysis result of IS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5885620, upper bound: 27.5611163
time: 54.22 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -37.0086174, 14.1792736, -37.0078468, 14.1792583, -51.0695496, 51.0612946
1: -19.7649708, 16.4556808, -19.7640800, 16.4555817, -36.2205505, 36.2197609
2: -13.6335983, 16.6161957, -13.6323032, 16.6156502, -29.6021614, 29.6029549
3: -13.9964409, 23.4906921, -13.9926701, 23.4900417, -37.0290070, 37.0424004
4: -18.6603355, 18.1593933, -18.6555519, 18.1586800, -36.8190155, 36.8149452
5: -16.1617699, 20.0274086, -16.1594067, 20.0267258, -36.1884956, 36.1868134
6: -25.9885597, 14.0093193, -25.9836922, 14.0095997, -39.9981613, 39.9930115
7: -23.3584156, 18.8888626, -23.3564110, 18.8885136, -42.2469292, 42.2452736
8: -20.7195282, 23.7516460, -20.7168427, 23.7514076, -44.3893433, 44.4004364
9: -14.7691889, 19.4869328, -14.7680798, 19.4863319, -34.2555199, 34.2550125
10: -29.7530823, 17.2089539, -29.7524261, 17.2036667, -46.9567490, 46.9613800
11: -33.8033371, 7.4749260, -33.8027420, 7.4739981, -41.2773361, 41.2776680
12: -27.9584236, 11.9459066, -27.9580116, 11.9442749, -39.4371033, 39.4419479
13: -18.1511230, 28.4966755, -18.1499538, 28.4962025, -46.6473236, 46.6466293
14: -56.6082191, -1.5102539, -56.6074753, -1.5121212, -49.9327927, 49.9772110
15: -21.8185921, 17.5917110, -21.8181953, 17.5897732, -39.4083633, 39.4099045
16: -33.0871696, 13.7833481, -33.0862732, 13.7783642, -46.8655319, 46.8696213
17: -62.9157181, 0.6895733, -62.9153976, 0.6882210, -62.0849457, 62.1496124
18: -34.8514328, 3.7437925, -34.8511887, 3.7422094, -36.8924637, 36.9138412
19: -27.3243084, 3.1802201, -27.3236599, 3.1786699, -30.5029793, 30.5038795
20: -19.1981564, 10.2027063, -19.1975155, 10.2009640, -28.7416840, 28.7332916
21: -31.7750587, 4.4114704, -31.7743244, 4.4106665, -36.1857262, 36.1857948
22: -32.2059250, 6.5903168, -32.2057724, 6.5891166, -38.4280701, 38.4359131
23: -23.4308586, 7.5362329, -23.4301586, 7.5337334, -30.9645920, 30.9663925
24: -28.0891991, 9.4654217, -28.0885544, 9.4629345, -37.5521317, 37.5539780
25: -22.0026817, 11.6411343, -22.0018654, 11.6390209, -33.5642090, 33.5668869
26: -34.9088669, 10.7746496, -34.9084854, 10.7728138, -43.7520370, 43.7797127
27: -28.7790604, 7.5821838, -28.7786293, 7.5811987, -36.3602600, 36.3608131
28: -22.4752312, 12.6431637, -22.4745750, 12.6408138, -35.1160431, 35.1177368
29: -34.4270782, 3.9496002, -34.4269180, 3.9486599, -38.3757401, 38.3765182
30: -25.9052162, 12.2333012, -25.9045086, 12.2323952, -38.1376114, 38.1378098
31: -34.2712212, 6.6308765, -34.2705841, 6.6291761, -40.9003983, 40.8994255
32: -20.7079163, 13.4517956, -20.7066536, 13.4505835, -34.1585007, 34.1584473
33: -30.1560669, 21.1909866, -30.1545982, 21.1902122, -51.1740265, 51.1616364
34: -28.8321724, 17.1590977, -28.8302650, 17.1581650, -45.9903374, 45.9893646
35: -25.9457684, 20.3011208, -25.9454575, 20.3002968, -46.2460632, 46.2465782
36: -24.5868816, 18.9847927, -24.5870113, 18.9844952, -43.5291061, 43.5280838
37: -44.7234154, 13.7926731, -44.7214737, 13.7914858, -58.2875977, 58.2652283
38: -33.0819435, 18.3402443, -33.0812683, 18.3396397, -51.4215851, 51.4215126
39: -34.7002335, 16.8218994, -34.6997337, 16.8211517, -51.3675537, 51.3660660
40: -34.6124344, 15.5752192, -34.6118889, 15.5749311, -49.6952515, 49.6920471
41: -24.5572929, 14.6827736, -24.5559235, 14.6826153, -39.2399063, 39.2386971
42: -16.4824524, 11.0945768, -16.4817200, 11.0914574, -27.5739098, 27.5762978

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=117, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 729
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 683

## Relational analysis of IS_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5216913, upper bound: 27.5571573
time: 48.34 seconds

## Relational analysis of IS_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5888584, upper bound: 27.5888583
time: 50.08 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 100.64 seconds
IS_A2_B2_B1_B1, status: Status.VERIFIED, split count: 4, time: 100.64
Output dim: 13, lower bound: -27.5216913, upper bound: 27.5611163
IS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 100.64
Output dim: 13, lower bound: -27.5885620, upper bound: 27.5611163
IS_A2_B2_B2_B1, status: Status.VERIFIED, split count: 4, time: 100.64
Output dim: 13, lower bound: -27.5216913, upper bound: 27.5571573
IS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 100.64
Output dim: 13, lower bound: -27.5888584, upper bound: 27.5888583

## BFS IS instance: IS_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -36.9852219, 14.1751986, -36.9441872, 14.1603088, -51.0298843, 50.9989700
1: -19.7343388, 16.4521599, -19.6984234, 16.4129562, -36.1472931, 36.1505814
2: -13.6031618, 16.6121883, -13.5680962, 16.5702724, -29.5283699, 29.5400734
3: -13.9479122, 23.4850311, -13.8914824, 23.4127502, -36.9101868, 36.9471741
4: -18.6234932, 18.1535072, -18.5777340, 18.0923138, -36.7158051, 36.7312393
5: -16.1146412, 20.0224762, -16.0620365, 19.9534378, -36.0680771, 36.0845108
6: -25.9769669, 14.0035000, -25.9432449, 14.0029650, -39.9799309, 39.9467468
7: -23.3184109, 18.8861694, -23.2722893, 18.8269749, -42.1453857, 42.1584587
8: -20.6809082, 23.7474861, -20.6345596, 23.6846962, -44.2882309, 44.3200760
9: -14.7433090, 19.4820480, -14.7092285, 19.4563828, -34.1996918, 34.1912766
10: -29.7393589, 17.2013550, -29.7234879, 17.1770210, -46.9163818, 46.9248428
11: -33.7911491, 7.4525471, -33.7401772, 7.4237943, -41.2149429, 41.1927261
12: -27.9531116, 11.9140034, -27.9043961, 11.8729906, -39.3768845, 39.3645859
13: -18.1126137, 28.4895477, -18.0669918, 28.4372311, -46.5498428, 46.5565414
14: -56.5953751, -1.5374565, -56.5441895, -1.5703907, -49.8868713, 49.8513222
15: -21.7938156, 17.5836983, -21.7630882, 17.5419121, -39.3357277, 39.3467865
16: -33.0649033, 13.7826548, -33.0285492, 13.7687473, -46.8336487, 46.8112030
17: -62.8991470, 0.6754951, -62.8653336, 0.6493206, -62.0344467, 62.0768433
18: -34.8443298, 3.7041807, -34.7963066, 3.6580687, -36.8295517, 36.8015099
19: -27.3173714, 3.1479745, -27.2601871, 3.1132865, -30.4306583, 30.4081612
20: -19.1921806, 10.1820126, -19.1715584, 10.1558876, -28.7034416, 28.6762581
21: -31.7651424, 4.3860869, -31.7172241, 4.3592620, -36.1244049, 36.1033096
22: -32.1985893, 6.5610199, -32.1488876, 6.5266180, -38.3650513, 38.3520584
23: -23.4240570, 7.4923067, -23.3577347, 7.4438267, -30.8678837, 30.8500404
24: -28.0834007, 9.4224348, -28.0193329, 9.3735809, -37.4569817, 37.4417686
25: -21.9965591, 11.6020403, -21.9438381, 11.5564194, -33.4808578, 33.4685860
26: -34.9018402, 10.7304230, -34.8536606, 10.6790962, -43.6785431, 43.6915970
27: -28.7713890, 7.5522003, -28.7224712, 7.5188470, -36.2902374, 36.2746735
28: -22.4695721, 12.6046391, -22.4142990, 12.5603371, -35.0299072, 35.0189362
29: -34.4177895, 3.9250317, -34.3632812, 3.8959732, -38.3137627, 38.2883148
30: -25.8978233, 12.2078743, -25.8567886, 12.1767311, -38.0745544, 38.0646629
31: -34.2630005, 6.5893917, -34.1919632, 6.5431566, -40.8061562, 40.7735825
32: -20.6972523, 13.4363270, -20.6547508, 13.4170952, -34.1143494, 34.0910797
33: -30.1446476, 21.1680069, -30.1078529, 21.1410179, -51.1014862, 51.0657120
34: -28.8260422, 17.1242065, -28.7790031, 17.0857201, -45.9117622, 45.9032097
35: -25.9380188, 20.2774429, -25.8977585, 20.2515297, -46.1895485, 46.1752014
36: -24.5808525, 18.9623203, -24.5426807, 18.9377232, -43.4755554, 43.4603729
37: -44.7073631, 13.7454739, -44.6215744, 13.6947479, -58.1654510, 58.0973129
38: -33.0715637, 18.3149338, -33.0269051, 18.2827244, -51.3542862, 51.3418388
39: -34.6842957, 16.7937508, -34.6255760, 16.7629414, -51.2923813, 51.2625694
40: -34.5990677, 15.5585194, -34.5587997, 15.5391884, -49.6447067, 49.6198578
41: -24.5432472, 14.6612387, -24.4913750, 14.6369896, -39.1802368, 39.1526146
42: -16.4721260, 11.0873594, -16.4428978, 11.0725155, -27.5446415, 27.5302582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=116, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 645
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 729
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1707

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 606

## Relational analysis of IS_A2_B2_B1_B2_B1

### Relational analysis result of IS_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5799827, upper bound: 27.5168961
time: 45.59 seconds

## Relational analysis of IS_A2_B2_B1_B2_B2

### Relational analysis result of IS_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5878909, upper bound: 27.5604434
time: 41.52 seconds

## BFS IS instance: IS_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -37.0078545, 14.1788740, -37.0057755, 14.1783018, -51.0717316, 51.0579910
1: -19.7643986, 16.4553757, -19.7625294, 16.4548492, -36.2192459, 36.2179031
2: -13.6332073, 16.6158028, -13.6312408, 16.6145935, -29.6034088, 29.6007919
3: -13.9961166, 23.4903049, -13.9918680, 23.4889908, -37.0350800, 37.0389824
4: -18.6601009, 18.1589165, -18.6549282, 18.1574173, -36.8175201, 36.8138428
5: -16.1614227, 20.0270920, -16.1584873, 20.0259247, -36.1873474, 36.1855774
6: -25.9845695, 14.0091171, -25.9732571, 14.0090580, -39.9936295, 39.9823761
7: -23.3573647, 18.8885822, -23.3536949, 18.8877811, -42.2451477, 42.2422791
8: -20.7187538, 23.7513161, -20.7149525, 23.7505455, -44.3922119, 44.3975372
9: -14.7686520, 19.4858017, -14.7666111, 19.4833336, -34.2519836, 34.2524109
10: -29.7522202, 17.2058048, -29.7502060, 17.1952801, -46.9475021, 46.9560089
11: -33.8020096, 7.4746714, -33.7993622, 7.4733419, -41.2753525, 41.2740326
12: -27.9580765, 11.9456139, -27.9570580, 11.9434996, -39.4332657, 39.4529648
13: -18.1506901, 28.4942360, -18.1488342, 28.4905014, -46.6411896, 46.6430702
14: -56.6074867, -1.5107574, -56.6055145, -1.5134125, -49.9256287, 49.9369888
15: -21.8181953, 17.5898647, -21.8172379, 17.5848713, -39.4030685, 39.4071045
16: -33.0860138, 13.7829256, -33.0832138, 13.7772579, -46.8632736, 46.8661385
17: -62.9147491, 0.6888981, -62.9128113, 0.6864471, -62.0822449, 62.1338501
18: -34.8494034, 3.7429867, -34.8456726, 3.7401428, -36.8883591, 36.8911133
19: -27.3237572, 3.1800375, -27.3221264, 3.1781807, -30.5019379, 30.5021629
20: -19.1974602, 10.2024984, -19.1956482, 10.2004671, -28.7404976, 28.7216301
21: -31.7742672, 4.4113002, -31.7722130, 4.4102001, -36.1844673, 36.1835136
22: -32.2052307, 6.5887489, -32.2039566, 6.5852976, -38.4215775, 38.4353180
23: -23.4302998, 7.5359254, -23.4285812, 7.5329866, -30.9632874, 30.9645061
24: -28.0884514, 9.4651670, -28.0865669, 9.4622889, -37.5507393, 37.5517349
25: -22.0022717, 11.6387482, -22.0007668, 11.6326809, -33.5554352, 33.5623169
26: -34.9069939, 10.7739124, -34.9035645, 10.7708912, -43.7474442, 43.7857018
27: -28.7762794, 7.5817471, -28.7714405, 7.5801148, -36.3563957, 36.3531876
28: -22.4745197, 12.6429424, -22.4727173, 12.6401653, -35.1146851, 35.1156616
29: -34.4261932, 3.9493256, -34.4245987, 3.9480095, -38.3742027, 38.3739243
30: -25.9044151, 12.2330990, -25.9024506, 12.2318506, -38.1362648, 38.1355515
31: -34.2705269, 6.6305676, -34.2687187, 6.6284313, -40.8989563, 40.8914871
32: -20.7043438, 13.4516268, -20.6971283, 13.4501152, -34.1544571, 34.1487541
33: -30.1555214, 21.1898632, -30.1531754, 21.1872864, -51.1592407, 51.1591263
34: -28.8316231, 17.1581688, -28.8288574, 17.1556396, -45.9872627, 45.9870262
35: -25.9452400, 20.3001251, -25.9441376, 20.2976208, -46.2428589, 46.2442627
36: -24.5862942, 18.9844627, -24.5854912, 18.9835968, -43.5268021, 43.5262375
37: -44.7225609, 13.7920551, -44.7192268, 13.7898731, -58.2756805, 58.2623291
38: -33.0783195, 18.3396454, -33.0714531, 18.3380585, -51.4163780, 51.4110985
39: -34.6993637, 16.8211327, -34.6974716, 16.8190823, -51.3637314, 51.3629913
40: -34.6114807, 15.5750217, -34.6093864, 15.5743551, -49.6920624, 49.6892853
41: -24.5532722, 14.6825342, -24.5451374, 14.6820126, -39.2352829, 39.2276726
42: -16.4812164, 11.0942650, -16.4784698, 11.0906219, -27.5718384, 27.5727348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=116, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 1707

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 606

## Relational analysis of IS_A2_B2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5808465, upper bound: 27.5452300
time: 40.67 seconds

## Relational analysis of IS_A2_B2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5233118, upper bound: 27.5565310
time: 207.51 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 250.37 seconds
IS_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 250.37
Output dim: 13, lower bound: -27.5799827, upper bound: 27.5168961
IS_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 250.37
Output dim: 13, lower bound: -27.5878909, upper bound: 27.5604434
IS_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 250.37
Output dim: 13, lower bound: -27.5808465, upper bound: 27.5452300
IS_A2_B2_B2_B2_B2, status: Status.VERIFIED, split count: 5, time: 250.37
Output dim: 13, lower bound: -27.5233118, upper bound: 27.5565310

## BFS IS instance: IS_A2_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -36.9741745, 14.1562462, -36.9172821, 14.1229649, -50.9728241, 50.9428787
1: -19.7286472, 16.4423409, -19.6918488, 16.3921928, -36.1208420, 36.1341896
2: -13.5968523, 16.6079121, -13.5544720, 16.5605965, -29.5088730, 29.5187683
3: -13.9336586, 23.4792366, -13.8612232, 23.3942757, -36.8783188, 36.9089165
4: -18.6169586, 18.1470928, -18.5637779, 18.0776634, -36.6946220, 36.7108688
5: -16.0991650, 20.0181007, -16.0293350, 19.9375877, -36.0367508, 36.0474358
6: -25.9564781, 13.9999552, -25.8981953, 13.9910336, -39.9475098, 39.8981514
7: -23.3135643, 18.8813839, -23.2672596, 18.8160591, -42.1296234, 42.1486435
8: -20.6725197, 23.7369728, -20.6177769, 23.6602669, -44.2527695, 44.2913513
9: -14.7356606, 19.4672222, -14.6950951, 19.4248924, -34.1605530, 34.1623154
10: -29.7341595, 17.1688919, -29.7004204, 17.1066933, -46.8408508, 46.8693123
11: -33.7856445, 7.4286003, -33.7191505, 7.3719373, -41.1575813, 41.1477509
12: -27.9438972, 11.9052162, -27.8832474, 11.8547354, -39.3427353, 39.3286095
13: -18.0660439, 28.4847450, -17.9647255, 28.4157505, -46.4817963, 46.4494705
14: -56.5872993, -1.5700893, -56.5091209, -1.6406078, -49.8105164, 49.7844772
15: -21.7882156, 17.5760231, -21.7511978, 17.5249882, -39.3132019, 39.3272209
16: -33.0535889, 13.7545376, -32.9966316, 13.7078733, -46.7614632, 46.7511673
17: -62.8936996, 0.6621552, -62.8542938, 0.6188126, -62.0086975, 62.0654984
18: -34.8406372, 3.6770897, -34.7793770, 3.5996819, -36.7641411, 36.7546387
19: -27.3125725, 3.1368990, -27.2444420, 3.0892754, -30.4018478, 30.3813400
20: -19.1885109, 10.1743774, -19.1628876, 10.1389885, -28.6803207, 28.6557846
21: -31.7589226, 4.3725138, -31.6970291, 4.3304424, -36.0893631, 36.0695419
22: -32.1915054, 6.5482988, -32.1261902, 6.4991479, -38.3290787, 38.3154831
23: -23.4191227, 7.4706898, -23.3384895, 7.3970447, -30.8161678, 30.8091793
24: -28.0776749, 9.3975563, -27.9933815, 9.3197985, -37.3974724, 37.3909378
25: -21.9904327, 11.5863495, -21.9246864, 11.5220308, -33.4381027, 33.4317093
26: -34.8954620, 10.7110167, -34.8362999, 10.6363001, -43.6267548, 43.6524773
27: -28.7650909, 7.5276132, -28.6974182, 7.4661908, -36.2312813, 36.2250328
28: -22.4661293, 12.5890493, -22.4025059, 12.5257034, -34.9918327, 34.9915543
29: -34.4103470, 3.9103279, -34.3380775, 3.8639088, -38.2742538, 38.2484055
30: -25.8919487, 12.1870937, -25.8363438, 12.1316319, -38.0235825, 38.0234375
31: -34.2560959, 6.5700798, -34.1669998, 6.5017424, -40.7578392, 40.7286682
32: -20.6868706, 13.4319134, -20.6300220, 13.4084301, -34.0952988, 34.0619354
33: -30.1075249, 21.1646004, -30.0264797, 21.1368637, -51.0637894, 50.9802017
34: -28.8187637, 17.1145744, -28.7613144, 17.0649433, -45.8837051, 45.8758888
35: -25.9067879, 20.2746086, -25.8293133, 20.2478867, -46.1546745, 46.1039200
36: -24.5473061, 18.9601650, -24.4698029, 18.9365501, -43.4401016, 43.3835373
37: -44.6844902, 13.7433548, -44.5696869, 13.6922722, -58.1395111, 58.0408401
38: -33.0330200, 18.3082924, -32.9445534, 18.2697639, -51.3027840, 51.2528458
39: -34.6396294, 16.7908745, -34.5260658, 16.7587395, -51.2418823, 51.1590195
40: -34.5825500, 15.5543737, -34.5220032, 15.5316143, -49.6193848, 49.5770721
41: -24.5300121, 14.6578875, -24.4612293, 14.6298275, -39.1598396, 39.1191177
42: -16.4630661, 11.0823641, -16.4232025, 11.0607080, -27.5237732, 27.5055656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=115, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 645
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1707

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 605

## Relational analysis of IS_A2_B2_B1_B2_B1_B1

### Relational analysis result of IS_A2_B2_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5233118, upper bound: 27.4787030
time: 59.99 seconds

## Relational analysis of IS_A2_B2_B1_B2_B1_B2

### Relational analysis result of IS_A2_B2_B1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5291366, upper bound: 27.4787030
time: 46.10 seconds

## BFS IS instance: IS_A2_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -36.9849930, 14.1746187, -36.9433823, 14.1587095, -51.0282516, 51.0022888
1: -19.7342243, 16.4513283, -19.6980362, 16.4099369, -36.1441612, 36.1493645
2: -13.6030025, 16.6120949, -13.5675735, 16.5699577, -29.5269966, 29.5374222
3: -13.9476957, 23.4849052, -13.8906775, 23.4122963, -36.9094925, 36.9375343
4: -18.6233139, 18.1533775, -18.5770016, 18.0918407, -36.7151566, 36.7303772
5: -16.1144009, 20.0223656, -16.0611801, 19.9530354, -36.0674362, 36.0835457
6: -25.9766808, 14.0034237, -25.9422092, 14.0026779, -39.9793587, 39.9456329
7: -23.3182564, 18.8858109, -23.2717552, 18.8256321, -42.1438904, 42.1575661
8: -20.6807556, 23.7471046, -20.6340027, 23.6833115, -44.2853317, 44.3161240
9: -14.7431860, 19.4818363, -14.7088032, 19.4556141, -34.1987991, 34.1906395
10: -29.7391815, 17.2009888, -29.7229538, 17.1757030, -46.9148865, 46.9239426
11: -33.7909355, 7.4522986, -33.7394333, 7.4228559, -41.2137909, 41.1917305
12: -27.9528885, 11.9138069, -27.9036045, 11.8723507, -39.3825760, 39.3624611
13: -18.1121330, 28.4894142, -18.0652771, 28.4367714, -46.5489044, 46.5546913
14: -56.5951843, -1.5377998, -56.5433350, -1.5715389, -49.8465271, 49.8500748
15: -21.7937012, 17.5835114, -21.7625751, 17.5411854, -39.3348846, 39.3460846
16: -33.0646782, 13.7823181, -33.0277328, 13.7675829, -46.8322601, 46.8100510
17: -62.8988953, 0.6751938, -62.8644485, 0.6482315, -62.0396805, 62.0748291
18: -34.8442307, 3.7038870, -34.7959442, 3.6570358, -36.8059311, 36.8008118
19: -27.3171902, 3.1478586, -27.2595673, 3.1128135, -30.4300041, 30.4074249
20: -19.1920586, 10.1819057, -19.1711121, 10.1555042, -28.6980629, 28.6739540
21: -31.7649422, 4.3859282, -31.7164841, 4.3586979, -36.1236420, 36.1024132
22: -32.1983833, 6.5608735, -32.1480675, 6.5261145, -38.3578644, 38.3511124
23: -23.4239159, 7.4920888, -23.3572254, 7.4429893, -30.8669052, 30.8493137
24: -28.0832367, 9.4221725, -28.0187283, 9.3726521, -37.4558868, 37.4409027
25: -21.9964142, 11.6018353, -21.9433136, 11.5557041, -33.4752045, 33.4678535
26: -34.9016418, 10.7302094, -34.8529358, 10.6782799, -43.6613998, 43.6875801
27: -28.7711983, 7.5519438, -28.7217369, 7.5179348, -36.2891312, 36.2736816
28: -22.4694481, 12.6044750, -22.4138451, 12.5597363, -35.0291824, 35.0183182
29: -34.4175072, 3.9248762, -34.3623161, 3.8954296, -38.3129349, 38.2871933
30: -25.8976097, 12.2076473, -25.8560238, 12.1759243, -38.0735321, 38.0636711
31: -34.2628136, 6.5891895, -34.1912613, 6.5423126, -40.8051262, 40.7726479
32: -20.6967678, 13.4362087, -20.6529999, 13.4166803, -34.1134491, 34.0892105
33: -30.1442165, 21.1679211, -30.1063728, 21.1406517, -51.1006317, 51.0598068
34: -28.8259335, 17.1240654, -28.7785416, 17.0852451, -45.9111786, 45.9026070
35: -25.9376755, 20.2773781, -25.8965950, 20.2513046, -46.1889801, 46.1739731
36: -24.5803604, 18.9622803, -24.5409107, 18.9375534, -43.4748459, 43.4575195
37: -44.7066422, 13.7454128, -44.6189919, 13.6944017, -58.1637421, 58.0985870
38: -33.0710526, 18.3147583, -33.0250778, 18.2821465, -51.3531990, 51.3398361
39: -34.6838417, 16.7936211, -34.6238556, 16.7625313, -51.2914886, 51.2605247
40: -34.5985947, 15.5584030, -34.5570564, 15.5388536, -49.6436081, 49.6177216
41: -24.5429268, 14.6611099, -24.4902401, 14.6364536, -39.1793823, 39.1513519
42: -16.4718933, 11.0871687, -16.4420815, 11.0718489, -27.5437431, 27.5292511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=115, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1707

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 605

## Relational analysis of IS_A2_B2_B1_B2_B2_B1

### Relational analysis result of IS_A2_B2_B1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5233118, upper bound: 27.4433208
time: 46.16 seconds

## Relational analysis of IS_A2_B2_B1_B2_B2_B2

### Relational analysis result of IS_A2_B2_B1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5361644, upper bound: 27.5087157
time: 113.85 seconds

## BFS IS instance: IS_A2_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -36.9968109, 14.1599073, -36.9787903, 14.1409168, -51.0145493, 51.0017395
1: -19.7586861, 16.4455757, -19.7558804, 16.4340458, -36.1927338, 36.2014542
2: -13.6269150, 16.6115284, -13.6175575, 16.6048813, -29.5838318, 29.5793533
3: -13.9818687, 23.4845123, -13.9615459, 23.4704857, -37.0032043, 37.0005875
4: -18.6535721, 18.1525116, -18.6409264, 18.1427307, -36.7963028, 36.7934380
5: -16.1459885, 20.0227261, -16.1257210, 20.0100975, -36.1560860, 36.1484451
6: -25.9641037, 14.0055637, -25.9282227, 13.9970989, -39.9612045, 39.9337845
7: -23.3525696, 18.8837986, -23.3485928, 18.8768578, -42.2294273, 42.2323914
8: -20.7103653, 23.7407875, -20.6981316, 23.7261009, -44.3567581, 44.3687515
9: -14.7610111, 19.4709854, -14.7524071, 19.4518471, -34.2128601, 34.2233925
10: -29.7470837, 17.1733322, -29.7270641, 17.1249390, -46.8720245, 46.9003983
11: -33.7965469, 7.4507327, -33.7781792, 7.4214511, -41.2179985, 41.2289124
12: -27.9488297, 11.9368315, -27.9358826, 11.9252014, -39.3989716, 39.4169807
13: -18.1041145, 28.4894562, -18.0465298, 28.4690361, -46.5731506, 46.5359879
14: -56.5994415, -1.5433788, -56.5704002, -1.5837135, -49.8494110, 49.8703613
15: -21.8125877, 17.5821991, -21.8052940, 17.5679092, -39.3804970, 39.3874931
16: -33.0746689, 13.7548342, -33.0512466, 13.7163563, -46.7910233, 46.8060799
17: -62.9093170, 0.6755905, -62.9015808, 0.6559048, -62.0568542, 62.1225052
18: -34.8457336, 3.7159128, -34.8287392, 3.6817131, -36.8227577, 36.8441849
19: -27.3189526, 3.1689277, -27.3063545, 3.1541381, -30.4730911, 30.4752827
20: -19.1938019, 10.1948404, -19.1869545, 10.1835213, -28.7172165, 28.7011414
21: -31.7680740, 4.3977165, -31.7519760, 4.3813562, -36.1494293, 36.1496925
22: -32.1981468, 6.5760422, -32.1812134, 6.5577502, -38.3855362, 38.3986206
23: -23.4253578, 7.5143003, -23.4093552, 7.4861770, -30.9115353, 30.9236565
24: -28.0827293, 9.4403000, -28.0606194, 9.4084911, -37.4912186, 37.5009193
25: -21.9961472, 11.6230431, -21.9816284, 11.5982666, -33.5125618, 33.5253906
26: -34.9006081, 10.7545023, -34.8860893, 10.7280493, -43.6955490, 43.7465248
27: -28.7699833, 7.5571499, -28.7463322, 7.5274458, -36.2974281, 36.3034821
28: -22.4710884, 12.6273260, -22.4609051, 12.6055002, -35.0765877, 35.0882301
29: -34.4187813, 3.9346285, -34.3992424, 3.9159412, -38.3347244, 38.3338699
30: -25.8985329, 12.2122974, -25.8819485, 12.1867657, -38.0852966, 38.0942459
31: -34.2636261, 6.6112413, -34.2437439, 6.5869942, -40.8506203, 40.8465614
32: -20.6939526, 13.4472151, -20.6723499, 13.4414434, -34.1353951, 34.1195641
33: -30.1184273, 21.1864395, -30.0718327, 21.1830788, -51.1215668, 51.0736389
34: -28.8243217, 17.1485558, -28.8111649, 17.1348190, -45.9591408, 45.9597206
35: -25.9140301, 20.2972813, -25.8756905, 20.2939796, -46.2080078, 46.1729736
36: -24.5527649, 18.9822788, -24.5125504, 18.9823952, -43.4913254, 43.4492950
37: -44.6996536, 13.7898808, -44.6672974, 13.7873192, -58.2498169, 58.2055206
38: -33.0397339, 18.3330326, -32.9890518, 18.3250122, -51.3647461, 51.3220825
39: -34.6547165, 16.8182774, -34.5979843, 16.8147888, -51.3131638, 51.2593002
40: -34.5948181, 15.5708809, -34.5724640, 15.5667248, -49.6667786, 49.6463013
41: -24.5400543, 14.6791630, -24.5149651, 14.6748409, -39.2148972, 39.1941299
42: -16.4721603, 11.0892811, -16.4586945, 11.0787954, -27.5509567, 27.5479755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=115, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 645
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 729
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1707

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 605

## Relational analysis of IS_A2_B2_B2_B2_B1_B1

### Relational analysis result of IS_A2_B2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5746402, upper bound: 27.5074984
time: 116.67 seconds

## Relational analysis of IS_A2_B2_B2_B2_B1_B2

### Relational analysis result of IS_A2_B2_B2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5303622, upper bound: 27.5159246
time: 42.63 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 161.51 seconds
IS_A2_B2_B1_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 161.51
Output dim: 13, lower bound: -27.5233118, upper bound: 27.4787030
IS_A2_B2_B1_B2_B1_B2, status: Status.VERIFIED, split count: 6, time: 161.51
Output dim: 13, lower bound: -27.5291366, upper bound: 27.4787030
IS_A2_B2_B1_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 161.51
Output dim: 13, lower bound: -27.5233118, upper bound: 27.4433208
IS_A2_B2_B1_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 161.51
Output dim: 13, lower bound: -27.5361644, upper bound: 27.5087157
IS_A2_B2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 161.51
Output dim: 13, lower bound: -27.5746402, upper bound: 27.5074984
IS_A2_B2_B2_B2_B1_B2, status: Status.VERIFIED, split count: 6, time: 161.51
Output dim: 13, lower bound: -27.5303622, upper bound: 27.5159246

## BFS IS instance: IS_A2_B2_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -36.9968109, 14.1599073, -36.9422684, 14.1191769, -50.9763794, 50.9596710
1: -19.7586861, 16.4455757, -19.7333736, 16.4241600, -36.1828461, 36.1789474
2: -13.6269150, 16.6115284, -13.5554829, 16.5835495, -29.5735245, 29.5162430
3: -13.9818687, 23.4845123, -13.8605499, 23.4316845, -36.9866409, 36.8984756
4: -18.6535721, 18.1525116, -18.5776558, 18.1168671, -36.7704391, 36.7301674
5: -16.1459885, 20.0227261, -16.0266171, 19.9755745, -36.1215630, 36.0493431
6: -25.9641037, 14.0055637, -25.8963337, 13.9843235, -39.9484253, 39.9018974
7: -23.3525696, 18.8837986, -23.2996292, 18.8604832, -42.2130508, 42.1834259
8: -20.7103653, 23.7407875, -20.6301861, 23.6960754, -44.3395233, 44.2992401
9: -14.7610111, 19.4709854, -14.7178955, 19.4244289, -34.1854401, 34.1888809
10: -29.7470837, 17.1733322, -29.6952820, 17.0402908, -46.7873764, 46.8686142
11: -33.7965469, 7.4507327, -33.7451935, 7.3332009, -41.1297493, 41.1959267
12: -27.9488297, 11.9368315, -27.9055653, 11.8437471, -39.3158722, 39.4224586
13: -18.1041145, 28.4894562, -17.8998585, 28.4323177, -46.5364304, 46.3893127
14: -56.5994415, -1.5433788, -56.5261536, -1.6786480, -49.7495575, 49.8877792
15: -21.8125877, 17.5821991, -21.7649040, 17.5513573, -39.3639450, 39.3471031
16: -33.0746689, 13.7548342, -33.0122299, 13.6483278, -46.7229958, 46.7670631
17: -62.9093170, 0.6755905, -62.8883209, 0.6200867, -62.0297318, 62.0731354
18: -34.8457336, 3.7159128, -34.8019409, 3.5726385, -36.7100906, 36.8656235
19: -27.3189526, 3.1689277, -27.2733898, 3.0909176, -30.4098701, 30.4423180
20: -19.1938019, 10.1948404, -19.1723633, 10.1323166, -28.6658287, 28.7045860
21: -31.7680740, 4.3977165, -31.7161503, 4.3175054, -36.0855789, 36.1138687
22: -32.1981468, 6.5760422, -32.1498184, 6.4951715, -38.3226700, 38.3819809
23: -23.4253578, 7.5143003, -23.3752022, 7.3944201, -30.8197784, 30.8895035
24: -28.0827293, 9.4403000, -28.0232277, 9.3112907, -37.3940201, 37.4635277
25: -21.9961472, 11.6230431, -21.9524803, 11.5237217, -33.4378281, 33.5086250
26: -34.9006081, 10.7545023, -34.8636398, 10.6213531, -43.5883789, 43.7669411
27: -28.7699833, 7.5571499, -28.7052288, 7.4256926, -36.1956749, 36.2623787
28: -22.4710884, 12.6273260, -22.4346752, 12.5274496, -34.9985390, 35.0620003
29: -34.4187813, 3.9346285, -34.3622589, 3.8521576, -38.2709389, 38.2968864
30: -25.8985329, 12.2122974, -25.8545895, 12.1197605, -38.0182953, 38.0668869
31: -34.2636261, 6.6112413, -34.1970940, 6.4992237, -40.7628479, 40.8083344
32: -20.6939526, 13.4472151, -20.6377335, 13.3986492, -34.0926018, 34.0849495
33: -30.1184273, 21.1864395, -29.9790306, 21.1825943, -51.1152649, 50.9879456
34: -28.8243217, 17.1485558, -28.7881508, 17.0800438, -45.9043655, 45.9367065
35: -25.9140301, 20.2972813, -25.8186913, 20.2919121, -46.2059402, 46.1159744
36: -24.5527649, 18.9822788, -24.4748821, 18.9596596, -43.4685516, 43.4117584
37: -44.6996536, 13.7898808, -44.6165924, 13.7294617, -58.1697922, 58.1515427
38: -33.0397339, 18.3330326, -32.9463081, 18.2657013, -51.3054352, 51.2793427
39: -34.6547165, 16.8182774, -34.5150528, 16.7819309, -51.2803116, 51.1753197
40: -34.5948181, 15.5708809, -34.5347939, 15.5373220, -49.6363220, 49.6087112
41: -24.5400543, 14.6791630, -24.4832211, 14.6270885, -39.1671448, 39.1623840
42: -16.4721603, 11.0892811, -16.4401283, 11.0480251, -27.5201855, 27.5294094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=114, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=333, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 645
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1707

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1719

## Relational analysis of IS_A2_B2_B2_B2_B1_B1_B1

### Relational analysis result of IS_A2_B2_B2_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.4909511, upper bound: 27.4385429
time: 42.08 seconds

## Relational analysis of IS_A2_B2_B2_B2_B1_B1_B2

### Relational analysis result of IS_A2_B2_B2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5734564, upper bound: 27.5063239
time: 62.06 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 106.36 seconds
IS_A2_B2_B2_B2_B1_B1_B1, status: Status.VERIFIED, split count: 7, time: 106.36
Output dim: 13, lower bound: -27.4909511, upper bound: 27.4385429
IS_A2_B2_B2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 106.36
Output dim: 13, lower bound: -27.5734564, upper bound: 27.5063239

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 108.63 + 1700.52 = 1809.15 seconds
