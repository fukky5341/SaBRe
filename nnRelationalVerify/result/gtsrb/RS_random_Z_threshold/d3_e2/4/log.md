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
execution time: IAR + RelationalAnalysis = 2.96 + 160.86 = 163.81 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -38.2196770, upper bound: 38.2196770

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 755

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 537

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1910028, upper bound: 38.1908164
time: 84.40 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1908164, upper bound: 38.1910028
time: 76.16 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 160.57 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 160.57
Output dim: 2, lower bound: -38.1910028, upper bound: 38.1908164
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 160.57
Output dim: 2, lower bound: -38.1908164, upper bound: 38.1910028

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
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8373871, 90.8373566
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
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1567764, 80.1567688
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0982895, 105.0982285
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2917328, 81.2916412
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6587372, 84.6586609
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885986, 86.1885986
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9701996, 107.9701538
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3349762, 87.3349686
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 563

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1624

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1711392, upper bound: 38.1817932
time: 83.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1819771, upper bound: 38.1709508
time: 185.24 seconds

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
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8373871, 90.8373871
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
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1568069, 80.1567841
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0983200, 105.0982895
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2917633, 81.2917328
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6587524, 84.6587372
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885986, 86.1885986
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9701691, 107.9701843
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3349915, 87.3349915
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1669

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 571

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1743759, upper bound: 38.1865190
time: 77.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1863416, upper bound: 38.1745639
time: 96.02 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 175.63 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 175.63
Output dim: 2, lower bound: -38.1711392, upper bound: 38.1817932
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 175.63
Output dim: 2, lower bound: -38.1819771, upper bound: 38.1709508
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 175.63
Output dim: 2, lower bound: -38.1743759, upper bound: 38.1865190
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 175.63
Output dim: 2, lower bound: -38.1863416, upper bound: 38.1745639

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
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8356476, 90.8376923
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
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1554031, 80.1563950
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0931625, 105.0972443
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2869110, 81.2933044
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6560822, 84.6604691
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885071, 86.1885757
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9685974, 107.9705429
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3335266, 87.3346329
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 758

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1696465, upper bound: 38.1757680
time: 103.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1650238, upper bound: 38.1803375
time: 153.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2

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
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8377228, 90.8356171
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
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1564102, 80.1553955
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0972977, 105.0931015
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2934113, 81.2868195
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6605377, 84.6560135
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885834, 86.1884995
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9705811, 107.9685669
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3346405, 87.3335037
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 548

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 522

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1814606, upper bound: 38.1706204
time: 79.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1816507, upper bound: 38.1704271
time: 89.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1

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
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8373871, 90.8373032
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
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1567917, 80.1567459
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0981979, 105.0980377
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2915955, 81.2913208
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6585846, 84.6584015
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885834, 86.1885910
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9701538, 107.9700699
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3349762, 87.3349152
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 672

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 984

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1274040, upper bound: 38.1862556
time: 114.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1741135, upper bound: 38.1394540
time: 88.10 seconds

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
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8373871, 90.8373871
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
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1568069, 80.1567841
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0983200, 105.0982056
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2917633, 81.2915802
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6587524, 84.6585846
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885986, 86.1885910
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9701691, 107.9701538
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3349915, 87.3349609
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 704

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 843

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1717240, upper bound: 38.1735527
time: 125.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1856405, upper bound: 38.1624470
time: 205.74 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 333.81 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 333.81
Output dim: 2, lower bound: -38.1696465, upper bound: 38.1757680
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 333.81
Output dim: 2, lower bound: -38.1650238, upper bound: 38.1803375
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 333.81
Output dim: 2, lower bound: -38.1814606, upper bound: 38.1706204
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 333.81
Output dim: 2, lower bound: -38.1816507, upper bound: 38.1704271
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 333.81
Output dim: 2, lower bound: -38.1274040, upper bound: 38.1862556
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 333.81
Output dim: 2, lower bound: -38.1741135, upper bound: 38.1394540
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 333.81
Output dim: 2, lower bound: -38.1717240, upper bound: 38.1735527
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 333.81
Output dim: 2, lower bound: -38.1856405, upper bound: 38.1624470

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

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
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8381805, 90.8360138
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
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1566620, 80.1556244
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0983658, 105.0940399
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2951050, 81.2883072
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6617279, 84.6570587
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885986, 86.1885223
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9710693, 107.9690018
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3349457, 87.3337555
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1713

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 706

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1810124, upper bound: 38.1439231
time: 87.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1548190, upper bound: 38.1701707
time: 101.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

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
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8381195, 90.8360748
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
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1566315, 80.1556549
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0982437, 105.0941772
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2948914, 81.2885208
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6615753, 84.6571960
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885986, 86.1885300
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9710083, 107.9690628
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3348999, 87.3337936
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 608

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1803566, upper bound: 38.1686343
time: 89.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1798753, upper bound: 38.1691119
time: 117.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

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
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8357544, 90.8351135
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
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1559906, 80.1556854
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0949173, 105.0936432
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2874985, 81.2855072
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6559601, 84.6545868
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885529, 86.1885300
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9685822, 107.9679871
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3340607, 87.3337250
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 966

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1575

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1228349, upper bound: 38.1858712
time: 83.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1270168, upper bound: 38.1817410
time: 80.90 seconds

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
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8302307, 90.8319550
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
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1530380, 80.1538696
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0841064, 105.0874252
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2864075, 81.2916183
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6615677, 84.6651154
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1884918, 86.1885452
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9627380, 107.9643555
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3312454, 87.3321533
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 762

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1720

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1837925, upper bound: 38.1364767
time: 121.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1598573, upper bound: 38.1605936
time: 88.24 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 211.73 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 211.73
Output dim: 2, lower bound: -38.1810124, upper bound: 38.1439231
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 211.73
Output dim: 2, lower bound: -38.1548190, upper bound: 38.1701707
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 211.73
Output dim: 2, lower bound: -38.1803566, upper bound: 38.1686343
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 211.73
Output dim: 2, lower bound: -38.1798753, upper bound: 38.1691119
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 211.73
Output dim: 2, lower bound: -38.1228349, upper bound: 38.1858712
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 211.73
Output dim: 2, lower bound: -38.1270168, upper bound: 38.1817410
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 211.73
Output dim: 2, lower bound: -38.1837925, upper bound: 38.1364767
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 211.73
Output dim: 2, lower bound: -38.1598573, upper bound: 38.1605936

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8353577, 90.8351440
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
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1551361, 80.1550293
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0949783, 105.0945358
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2907867, 81.2901154
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6548157, 84.6543579
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1884995, 86.1884766
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9689941, 107.9687881
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3344269, 87.3343048
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1720

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1705

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1200097, upper bound: 38.1820376
time: 87.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1189641, upper bound: 38.1830797
time: 165.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8357849, 90.8347168
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
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1553497, 80.1548233
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0958176, 105.0936966
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2920990, 81.2887878
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6557312, 84.6534576
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1884995, 86.1884613
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9693909, 107.9683838
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3346405, 87.3340759
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 893

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 753

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1125577, upper bound: 38.1668828
time: 72.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1121629, upper bound: 38.1672649
time: 144.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8291016, 90.8312912
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
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1525803, 80.1536255
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0812683, 105.0855331
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2821808, 81.2888412
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6595078, 84.6640625
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1884613, 86.1885376
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9612732, 107.9633484
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3304214, 87.3315735
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1643

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 626

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1638681, upper bound: 38.1361493
time: 101.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1834688, upper bound: 38.1163538
time: 92.06 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 195.34 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 195.34
Output dim: 2, lower bound: -38.1200097, upper bound: 38.1820376
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 195.34
Output dim: 2, lower bound: -38.1189641, upper bound: 38.1830797
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 195.34
Output dim: 2, lower bound: -38.1125577, upper bound: 38.1668828
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 195.34
Output dim: 2, lower bound: -38.1121629, upper bound: 38.1672649
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 195.34
Output dim: 2, lower bound: -38.1638681, upper bound: 38.1361493
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 195.34
Output dim: 2, lower bound: -38.1834688, upper bound: 38.1163538

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8298569, 90.8288727
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
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1525650, 80.1520691
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0931015, 105.0911331
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2896347, 81.2865601
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6521683, 84.6500702
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1883698, 86.1883240
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9678116, 107.9668655
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3345184, 87.3339844
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.0824243, upper bound: 38.1794471
time: 93.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1173707, upper bound: 38.1445896
time: 97.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8290939, 90.8296356
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
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1521683, 80.1524429
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0915604, 105.0926666
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2872238, 81.2889557
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6505203, 84.6517181
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1883392, 86.1883469
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9670792, 107.9675980
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3340912, 87.3343964
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1656

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1674

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1100284, upper bound: 38.1760158
time: 71.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1118419, upper bound: 38.1742414
time: 1160.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8335648, 90.8349915
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
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1525574, 80.1532135
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0919113, 105.0946655
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2950287, 81.2992935
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6709671, 84.6738739
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1886597, 86.1886978
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9663544, 107.9676971
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3289337, 87.3296814
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 609

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1560

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1600005, upper bound: 38.0937457
time: 98.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1612345, upper bound: 38.0925037
time: 71.59 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 172.37 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 172.37
Output dim: 2, lower bound: -38.0824243, upper bound: 38.1794471
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 172.37
Output dim: 2, lower bound: -38.1173707, upper bound: 38.1445896
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 172.37
Output dim: 2, lower bound: -38.1100284, upper bound: 38.1760158
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 172.37
Output dim: 2, lower bound: -38.1118419, upper bound: 38.1742414
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 172.37
Output dim: 2, lower bound: -38.1600005, upper bound: 38.0937457
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 172.37
Output dim: 2, lower bound: -38.1612345, upper bound: 38.0925037

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 163.81 + 4625.59 = 4789.40 seconds
