## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 4)
Time budget: 7200 seconds
Split limit: 100
Threshold: 38.181457323


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388)
1: (-26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461)
2: (-24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001)
3: (-28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680)
4: (-31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228)
5: (-28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674)
6: (-55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035)
7: (-32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505)
8: (-37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959)
9: (-30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573)
10: (-49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753)
11: (-48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688)
12: (-59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8373871, 90.8373871)
13: (-51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880)
14: (-79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897)
15: (-38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579)
16: (-48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783)
17: (-79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988)
18: (-48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538)
19: (-38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658)
20: (-34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885)
21: (-46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846)
22: (-49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286)
23: (-37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486)
24: (-45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513)
25: (-39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552)
26: (-56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825)
27: (-46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792)
28: (-37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667)
29: (-51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515)
30: (-46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646)
31: (-49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272)
32: (-55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1568069, 80.1567993)
33: (-73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0983200, 105.0983200)
34: (-63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2917633, 81.2917480)
35: (-60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6587524, 84.6587524)
36: (-60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885986, 86.1885986)
37: (-89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9701691, 107.9701843)
38: (-69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888)
39: (-83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197)
40: (-65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3349915, 87.3349915)
41: (-58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127)
42: (-40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.98 + 163.00 = 165.98 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -38.2196770, upper bound: 38.2196770

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1335854, upper bound: 38.2164821
time: 79.48 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2164820, upper bound: 38.1335854
time: 99.17 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 178.78 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 178.78
Output dim: 2, lower bound: -38.1335854, upper bound: 38.2164821
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 178.78
Output dim: 2, lower bound: -38.2164820, upper bound: 38.1335854

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8395462, 90.8396606
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1577988, 80.1578522
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0924759, 105.0926819
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2831268, 81.2834625
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6524811, 84.6527252
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885376, 86.1885452
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9709778, 107.9710770
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3343887, 87.3344421
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0599758, upper bound: 38.2089764
time: 92.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1262560, upper bound: 38.1434665
time: 79.03 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8396683, 90.8395538
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1578445, 80.1577988
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0926895, 105.0924683
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2834625, 81.2831268
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6527252, 84.6524887
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885376, 86.1885452
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9710846, 107.9709702
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3344498, 87.3343887
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1434665, upper bound: 38.1262560
time: 92.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2089764, upper bound: 38.0599758
time: 152.55 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 247.83 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 247.83
Output dim: 2, lower bound: -38.0599758, upper bound: 38.2089764
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 247.83
Output dim: 2, lower bound: -38.1262560, upper bound: 38.1434665
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 247.83
Output dim: 2, lower bound: -38.1434665, upper bound: 38.1262560
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 247.83
Output dim: 2, lower bound: -38.2089764, upper bound: 38.0599758

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8393936, 90.8396759
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1573257, 80.1574631
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0894394, 105.0900116
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2786713, 81.2795563
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6501923, 84.6507950
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885071, 86.1885300
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9706421, 107.9708939
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3339005, 87.3340454
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9879141, upper bound: 38.2062207
time: 95.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.0545660, upper bound: 38.1051677
time: 94.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8396683, 90.8393860
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1578445, 80.1573334
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0926895, 105.0894470
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2834625, 81.2786713
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6527252, 84.6501846
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885376, 86.1885147
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9710846, 107.9706268
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3344498, 87.3338928
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1051677, upper bound: 38.0545661
time: 88.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2062207, upper bound: 37.9879141
time: 95.09 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 186.51 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 186.51
Output dim: 2, lower bound: -37.9879141, upper bound: 38.2062207
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 186.51
Output dim: 2, lower bound: -38.0545660, upper bound: 38.1051677
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 186.51
Output dim: 2, lower bound: -38.1051677, upper bound: 38.0545661
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 186.51
Output dim: 2, lower bound: -38.2062207, upper bound: 37.9879141

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8452148, 90.8458862
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1602783, 80.1605988
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0726013, 105.0739441
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2601166, 81.2622223
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6361389, 84.6375961
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1886368, 86.1886673
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9763947, 107.9770203
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3338394, 87.3342133
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9395699, upper bound: 38.1993650
time: 93.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -37.9762120, upper bound: 38.1411181
time: 151.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8458862, 90.8452148
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1609650, 80.1602707
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0766144, 105.0726013
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2661285, 81.2601166
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6395264, 84.6361542
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1886826, 86.1886368
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9771881, 107.9763794
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3346176, 87.3338470
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.0393533, upper bound: 37.9762120
time: 119.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1993650, upper bound: 37.9395699
time: 83.11 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 205.38 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 205.38
Output dim: 2, lower bound: -37.9395699, upper bound: 38.1993650
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 205.38
Output dim: 2, lower bound: -37.9762120, upper bound: 38.1411181
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 205.38
Output dim: 2, lower bound: -38.0393533, upper bound: 37.9762120
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 205.38
Output dim: 2, lower bound: -38.1993650, upper bound: 37.9395699

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8450012, 90.8460541
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1593170, 80.1598206
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0678406, 105.0699005
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2522888, 81.2555313
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6313782, 84.6336060
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885605, 86.1885986
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9754486, 107.9764481
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3324966, 87.3330460
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 664

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -37.9098401, upper bound: 38.1158447
time: 119.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -37.8676510, upper bound: 38.1663128
time: 80.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8460388, 90.8450165
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1602020, 80.1593323
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0725861, 105.0678253
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2594147, 81.2522888
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6353760, 84.6313782
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1886368, 86.1885529
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9766083, 107.9754562
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3334427, 87.3324814
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 664

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1663128, upper bound: 37.8676510
time: 183.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1158447, upper bound: 37.9098401
time: 115.80 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 301.52 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 301.52
Output dim: 2, lower bound: -37.9098401, upper bound: 38.1158447
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 301.52
Output dim: 2, lower bound: -37.8676510, upper bound: 38.1663128
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 301.52
Output dim: 2, lower bound: -38.1663128, upper bound: 37.8676510
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 301.52
Output dim: 2, lower bound: -38.1158447, upper bound: 37.9098401

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 165.98 + 1935.84 = 2101.82 seconds
