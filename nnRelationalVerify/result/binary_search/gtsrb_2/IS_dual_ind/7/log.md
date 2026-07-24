## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 18000 seconds
Threshold: 97.2844837351
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716)
1: (-70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341)
2: (-63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282)
3: (-72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957)
4: (-76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267)
5: (-68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710)
6: (-102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202)
7: (-84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920)
8: (-89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154)
9: (-78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873)
10: (-111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498)
11: (-111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485)
12: (-111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295)
13: (-110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117)
14: (-163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874)
15: (-92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756)
16: (-118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149)
17: (-164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765)
18: (-102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608)
19: (-85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756)
20: (-74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135)
21: (-104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721)
22: (-113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161)
23: (-86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248)
24: (-103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021)
25: (-91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324)
26: (-122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165)
27: (-104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583)
28: (-85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112)
29: (-119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958)
30: (-102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372)
31: (-106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931)
32: (-100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959)
33: (-141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360)
34: (-120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018)
35: (-120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321)
36: (-117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379)
37: (-164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464)
38: (-145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569)
39: (-168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181)
40: (-135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575)
41: (-100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632)
42: (-75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712)

## BASE Result
execution time: IAR + LP analysis = 2.65 + 144.12 = 146.77 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -107.9309805, upper bound: 107.9309806


# Binary Search by BASE starts (time budget: 17853.23 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=159.03338623046875
rel_dist={5: [-102.21084835141579, 102.21084836698037]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=159.03338623046875
rel_dist={5: [-97.30393826817422, 97.30393825827855]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=159.03338623046875
rel_dist={5: [-92.81938600404467, 92.81938600410132]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=159.03338623046875
rel_dist={5: [-95.19916083299321, 95.19916083562723]}

## Binary Search Result
Binary search time: 634.21 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 17219.02 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3725448, upper bound: 103.4551319
time: 105.29 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3725448, upper bound: 103.4588061
time: 171.00 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 276.43 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 276.43
Output dim: 5, lower bound: -103.3725448, upper bound: 103.4551319
IS_A2, status: Status.UNKNOWN, split count: 1, time: 276.43
Output dim: 5, lower bound: -103.3725448, upper bound: 103.4588061

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -124.6066666, 84.2873840, -125.1241455, 84.4858322, -209.0924988, 209.4115295
1: -69.9321289, 74.2248077, -70.2930832, 74.3892517, -144.3213806, 144.5178680
2: -62.7293396, 71.1545944, -63.1982193, 71.3931427, -134.1224823, 134.3528137
3: -72.1667099, 86.0914154, -72.7509155, 86.4327774, -158.5994873, 158.8423157
4: -75.2612152, 84.4494247, -75.8243408, 84.6996765, -159.9608917, 160.2737427
5: -67.5027466, 90.4818497, -67.9779282, 90.8007507, -158.3034973, 158.4597778
6: -102.5512543, 75.7055969, -102.7784576, 76.0357971, -178.5870514, 178.4840546
7: -83.4754562, 91.1614227, -83.9013519, 91.3330917, -174.8085327, 175.0627747
8: -88.5101929, 101.5078583, -89.0208817, 101.7964935, -190.3066864, 190.5287323
9: -78.1663818, 81.6335449, -78.4707489, 81.9048615, -160.0712280, 160.1042938
10: -110.7386780, 117.2646561, -111.2959213, 118.2475739, -228.9862366, 228.5605774
11: -110.5426636, 83.1887283, -111.0318298, 84.0980835, -194.6407471, 194.2205505
12: -110.9254379, 88.7152863, -111.3575439, 89.5671692, -200.4926147, 200.0728302
13: -109.8729401, 100.1326675, -110.5012589, 100.5967026, -210.4696350, 210.6339111
14: -162.6055756, 83.4861450, -163.1429749, 84.2331390, -246.8386993, 246.6291199
15: -91.4326019, 81.5452423, -91.9494553, 81.7305450, -173.1631165, 173.4946899
16: -118.0348358, 97.1803436, -118.4019165, 97.7246704, -215.7595062, 215.5822601
17: -164.1159363, 119.0831833, -164.6089478, 120.1631165, -284.2790527, 283.6921387
18: -101.4866257, 84.3787079, -101.9237137, 85.1276398, -186.6142273, 186.3024292
19: -84.9405899, 47.4515610, -85.2985077, 47.8627930, -132.8033752, 132.7500610
20: -74.5630646, 57.3862228, -74.8781815, 57.7334785, -132.2965393, 132.2644043
21: -104.2998657, 62.9968300, -104.7278290, 63.5930176, -167.8928833, 167.7246399
22: -113.0589981, 72.8634033, -113.3402481, 73.3542099, -186.4132080, 186.2036438
23: -86.2505188, 58.2469292, -86.5473633, 58.7029724, -144.9534912, 144.7942810
24: -103.3558960, 69.1258240, -103.6671371, 69.4895554, -172.8454437, 172.7929688
25: -90.8022842, 67.9390488, -91.0389862, 68.3084412, -159.1107178, 158.9780273
26: -121.9121246, 89.4086609, -122.3752823, 90.2145157, -212.1266022, 211.7839355
27: -104.2366867, 73.9641800, -104.5433350, 74.3229980, -178.5596771, 178.5075073
28: -85.4963226, 63.0298767, -85.7359772, 63.3157883, -148.8121033, 148.7658539
29: -119.1377106, 76.5448532, -119.4069138, 77.1587524, -196.2964478, 195.9517670
30: -102.5404358, 79.3659515, -102.8659210, 79.9679489, -182.5083923, 182.2318726
31: -106.1160583, 66.8169556, -106.5710220, 67.3524017, -173.4684601, 173.3879700
32: -99.8150787, 73.3589783, -100.0820160, 73.6466522, -173.4617310, 173.4409790
33: -140.3820343, 80.5549545, -140.9797668, 80.8732986, -221.2553406, 221.5347290
34: -119.5900650, 72.7058487, -120.0369492, 72.9638519, -192.5539246, 192.7427979
35: -120.0107956, 70.1563034, -120.5726242, 70.3953018, -190.4060822, 190.7289276
36: -117.3122711, 69.6056137, -117.7717667, 69.7870483, -187.0993042, 187.3773804
37: -164.3631897, 73.8990097, -164.7310791, 74.1818466, -238.5450439, 238.6300659
38: -145.1600647, 86.1004486, -145.7359619, 86.3929291, -231.5529785, 231.8363953
39: -167.7764587, 77.8355713, -168.3667603, 78.0620422, -245.8384857, 246.2023315
40: -135.0021057, 73.6946869, -135.4476318, 73.8711853, -208.8732910, 209.1423035
41: -100.4634705, 67.0853043, -100.7287598, 67.3727417, -167.8362122, 167.8140564
42: -75.5640106, 65.2241974, -75.7875214, 65.7701263, -141.3341064, 141.0117035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3669610, upper bound: 103.3791810
time: 131.57 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3669610, upper bound: 103.4505855
time: 140.11 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -125.2836914, 84.5387802, -125.3210907, 84.5513153, -209.8350067, 209.8598633
1: -70.4095001, 74.4239502, -70.4336395, 74.4349594, -144.8444519, 144.8575897
2: -63.3581696, 71.4284592, -63.3880463, 71.4375610, -134.7957153, 134.8164978
3: -72.9513702, 86.4834442, -72.9886780, 86.4980164, -159.4493713, 159.4721222
4: -76.0163956, 84.7425385, -76.0504761, 84.7566528, -160.7730408, 160.7930145
5: -68.1348724, 90.8475342, -68.1660233, 90.8590546, -158.9939270, 159.0135498
6: -102.8498077, 76.0811539, -102.8711243, 76.1557541, -179.0055542, 178.9522705
7: -84.0359802, 91.3690567, -84.0659866, 91.3818130, -175.4177856, 175.4350433
8: -89.1975403, 101.8446808, -89.2292175, 101.8587418, -191.0562744, 191.0738983
9: -78.5486755, 81.9712906, -78.5812683, 82.0053482, -160.5540161, 160.5525513
10: -111.3890762, 118.5826645, -111.4086304, 118.6454010, -230.0344696, 229.9913025
11: -111.1012955, 84.4298019, -111.1205063, 84.4821320, -195.5834045, 195.5503082
12: -111.4130249, 89.8627014, -111.4298401, 89.9118576, -201.3248901, 201.2925415
13: -110.7405396, 100.6908951, -110.7688904, 100.7183609, -211.4588623, 211.4597778
14: -163.2510681, 84.5008774, -163.2775574, 84.5419006, -247.7929382, 247.7784271
15: -92.0728302, 81.7929382, -92.1443787, 81.8121796, -173.8850098, 173.9373016
16: -118.5161896, 97.8834991, -118.5435715, 97.9459229, -216.4620972, 216.4270630
17: -164.6907959, 120.5500946, -164.7074890, 120.6127243, -285.3035278, 285.2575684
18: -102.0187836, 85.3914032, -102.0448685, 85.4352264, -187.4539795, 187.4362793
19: -85.3569641, 48.0078011, -85.3700867, 48.0345497, -133.3915100, 133.3778839
20: -74.9406281, 57.8567352, -74.9569702, 57.8766174, -132.8172302, 132.8137054
21: -104.7878342, 63.8062553, -104.8059692, 63.8401718, -168.6280060, 168.6122131
22: -113.3832321, 73.5227127, -113.4259262, 73.5564575, -186.9396820, 186.9486389
23: -86.5999985, 58.8614655, -86.6124573, 58.8888512, -145.4888458, 145.4739227
24: -103.7299271, 69.6165771, -103.7512970, 69.6393051, -173.3692322, 173.3678589
25: -91.0931015, 68.4360886, -91.1056671, 68.4613190, -159.5543976, 159.5417480
26: -122.4458618, 90.4897308, -122.4702835, 90.5371628, -212.9829865, 212.9599915
27: -104.6329498, 74.4486694, -104.6592484, 74.4699173, -179.1028748, 179.1079102
28: -85.7934875, 63.4117355, -85.8058319, 63.4299240, -149.2234039, 149.2175598
29: -119.4589386, 77.3727341, -119.4809723, 77.4131165, -196.8720551, 196.8536987
30: -102.9253006, 80.1761322, -102.9425964, 80.2114105, -183.1366882, 183.1187286
31: -106.6536560, 67.5429840, -106.6733093, 67.5773392, -174.2309875, 174.2162781
32: -100.1512756, 73.7404404, -100.1758194, 73.7614441, -173.9127045, 173.9162598
33: -141.1815643, 80.9286804, -141.2188568, 80.9430237, -222.1245728, 222.1475220
34: -120.1827469, 73.0269165, -120.2116470, 73.0464020, -193.2291565, 193.2385559
35: -120.7642517, 70.4415741, -120.7998276, 70.4521484, -191.2163849, 191.2413788
36: -117.9226379, 69.8307648, -117.9573593, 69.8425064, -187.7651367, 187.7881165
37: -164.8338623, 74.2456055, -164.8668823, 74.2810974, -239.1149597, 239.1124878
38: -145.9229126, 86.4525146, -145.9647217, 86.4653549, -232.3882446, 232.4172363
39: -168.5646667, 78.1052399, -168.6044617, 78.1166229, -246.6812897, 246.7096863
40: -135.5876770, 73.8635330, -135.6213074, 73.9132767, -209.5009460, 209.4848328
41: -100.8066711, 67.4122086, -100.8266525, 67.4671097, -168.2737427, 168.2388611
42: -75.8448944, 65.9455109, -75.8609695, 65.9859314, -141.8308258, 141.8064880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.4543716, upper bound: 103.3837866
time: 120.68 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3669610, upper bound: 103.4543715
time: 111.50 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 234.36 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 234.36
Output dim: 5, lower bound: -103.3669610, upper bound: 103.3791810
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 234.36
Output dim: 5, lower bound: -103.3669610, upper bound: 103.4505855
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 234.36
Output dim: 5, lower bound: -103.4543716, upper bound: 103.3837866
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 234.36
Output dim: 5, lower bound: -103.3669610, upper bound: 103.4543715

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -124.5668335, 84.2766876, -124.8190155, 84.3986664, -208.9654999, 209.0957031
1: -69.9056091, 74.2174072, -70.0869904, 74.3317719, -144.2373810, 144.3043976
2: -62.6876945, 71.1471558, -62.8793678, 71.3356934, -134.0233765, 134.0265198
3: -72.1185760, 86.0810547, -72.3804779, 86.3526001, -158.4711761, 158.4615326
4: -75.2139816, 84.4407883, -75.4691467, 84.6266861, -159.8406677, 159.9099274
5: -67.4600754, 90.4726868, -67.6560364, 90.7298737, -158.1899414, 158.1287079
6: -102.5364456, 75.6800842, -102.6578598, 75.8419495, -178.3783875, 178.3379517
7: -83.4398727, 91.1535339, -83.6241150, 91.2723160, -174.7121887, 174.7776337
8: -88.4679184, 101.4978485, -88.6948547, 101.7195129, -190.1874084, 190.1926880
9: -78.1421661, 81.6159592, -78.2680817, 81.7691498, -159.9113007, 159.8840332
10: -110.7194901, 117.2064896, -111.1342468, 117.8106232, -228.5300903, 228.3407288
11: -110.5263824, 83.1119690, -110.9064331, 83.5241318, -194.0505066, 194.0184021
12: -110.9135132, 88.6437836, -111.2605209, 89.0328751, -199.9463806, 199.9042969
13: -109.8293686, 100.1100235, -110.1639252, 100.4233398, -210.2527161, 210.2739258
14: -162.5801392, 83.4290619, -162.9473877, 83.7901230, -246.3702698, 246.3764496
15: -91.3988113, 81.5303955, -91.6846924, 81.6115570, -173.0103455, 173.2150879
16: -118.0099945, 97.1457901, -118.1998978, 97.4676437, -215.4776306, 215.3456879
17: -164.0960999, 118.9943848, -164.4555664, 119.4735794, -283.5696716, 283.4499512
18: -101.4657440, 84.3208771, -101.7613068, 84.6616974, -186.1274414, 186.0821838
19: -84.9276276, 47.4160233, -85.1998520, 47.5863342, -132.5139618, 132.6158752
20: -74.5491943, 57.3569031, -74.7711029, 57.5035210, -132.0527191, 132.1279907
21: -104.2854691, 62.9426880, -104.6169434, 63.1818581, -167.4673309, 167.5596313
22: -113.0444260, 72.8172989, -113.2271118, 72.9823303, -186.0267639, 186.0443878
23: -86.2385864, 58.2120476, -86.4568329, 58.4365959, -144.6751862, 144.6688538
24: -103.3416595, 69.0970764, -103.5598297, 69.2537918, -172.5954590, 172.6568909
25: -90.7907791, 67.9102783, -90.9515152, 68.0719452, -158.8627319, 158.8617859
26: -121.8960114, 89.3375092, -122.2497025, 89.6693420, -211.5653534, 211.5871887
27: -104.2161865, 73.9266281, -104.3871765, 74.0213470, -178.2375336, 178.3137817
28: -85.4835510, 63.0020561, -85.6385803, 63.0921021, -148.5756531, 148.6406403
29: -119.1245651, 76.4811554, -119.3059540, 76.6675568, -195.7920990, 195.7870789
30: -102.5269623, 79.3147354, -102.7627563, 79.5738068, -182.1007690, 182.0774841
31: -106.0976410, 66.7811966, -106.4333496, 67.0625076, -173.1601562, 173.2145386
32: -99.7998734, 73.3290482, -99.9580154, 73.4235306, -173.2234039, 173.2870636
33: -140.3361053, 80.5409241, -140.6326599, 80.7622452, -221.0983582, 221.1735840
34: -119.5616226, 72.6889343, -119.8195114, 72.8224030, -192.3840332, 192.5084381
35: -119.9765854, 70.1467285, -120.3191605, 70.3147430, -190.2913208, 190.4658813
36: -117.2904968, 69.5940704, -117.6062317, 69.6914368, -186.9819183, 187.2003021
37: -164.3428345, 73.8725815, -164.5696716, 73.9822464, -238.3250732, 238.4422455
38: -145.1237793, 86.0865479, -145.4573364, 86.2785950, -231.4023438, 231.5438843
39: -167.7356567, 77.8251190, -168.0573425, 77.9828491, -245.7185059, 245.8824615
40: -134.9778900, 73.6815338, -135.2504883, 73.7717819, -208.7496643, 208.9320068
41: -100.4479752, 67.0602188, -100.5984802, 67.1875381, -167.6355133, 167.6586914
42: -75.5531311, 65.1823730, -75.6871414, 65.4585648, -141.0116882, 140.8695068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=678, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 647

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3440903, upper bound: 103.2849830
time: 120.13 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3440903, upper bound: 103.3578408
time: 108.01 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -124.5885315, 84.2806091, -125.1543884, 84.5831604, -209.1716919, 209.4349976
1: -69.9194565, 74.2206497, -70.3031845, 74.4772339, -144.3966827, 144.5238342
2: -62.7115135, 71.1501160, -63.1819229, 71.6453857, -134.3569031, 134.3320312
3: -72.1470261, 86.0845795, -72.7354050, 86.7337875, -158.8808136, 158.8199768
4: -75.2407684, 84.4438934, -75.8122025, 84.9081955, -160.1489563, 160.2560883
5: -67.4856110, 90.4760132, -67.9727173, 91.1225433, -158.6081543, 158.4487152
6: -102.5418091, 75.6609879, -102.8538818, 75.9846497, -178.5264587, 178.5148621
7: -83.4568405, 91.1561661, -83.9108124, 91.4411774, -174.8980103, 175.0669861
8: -88.4919205, 101.5011978, -89.0127106, 102.0217209, -190.5136108, 190.5139160
9: -78.1542358, 81.6179962, -78.4935608, 81.9441376, -160.0983734, 160.1115570
10: -110.7288742, 117.2382050, -111.4526749, 118.2846451, -229.0134888, 228.6908569
11: -110.5288544, 83.1589508, -111.4112091, 84.0543976, -194.5832520, 194.5701599
12: -110.9181595, 88.6862488, -111.7835236, 89.5514984, -200.4696655, 200.4697571
13: -109.8304901, 100.1178970, -110.4446030, 100.8080292, -210.6385193, 210.5624847
14: -162.5919647, 83.4654388, -163.3832245, 84.2059021, -246.7978668, 246.8486633
15: -91.3733978, 81.5340881, -91.9061584, 81.8098297, -173.1832275, 173.4402466
16: -118.0199432, 97.1165237, -118.5236053, 97.6380768, -215.6580200, 215.6401062
17: -164.1058350, 119.0468979, -164.9842224, 120.1034241, -284.2092590, 284.0311279
18: -101.4741287, 84.3559036, -102.2045288, 85.1001129, -186.5742188, 186.5604248
19: -84.9330063, 47.4374657, -85.5652924, 47.8446617, -132.7776642, 133.0027618
20: -74.5541382, 57.3750648, -75.0501099, 57.7268906, -132.2810364, 132.4251404
21: -104.2900772, 62.9776764, -105.0737610, 63.5666542, -167.8567200, 168.0514221
22: -113.0488892, 72.8425446, -113.4565887, 73.3308411, -186.3797302, 186.2991180
23: -86.2444305, 58.2328911, -86.7583160, 58.6969986, -144.9414215, 144.9911957
24: -103.3464966, 69.1161194, -103.8167343, 69.4791412, -172.8256378, 172.9328613
25: -90.7942657, 67.9245758, -91.1177216, 68.2896500, -159.0839081, 159.0422974
26: -121.9006577, 89.3816376, -122.7851410, 90.1920471, -212.0926819, 212.1667633
27: -104.2236862, 73.9512482, -104.7167511, 74.3061295, -178.5298157, 178.6679993
28: -85.4900894, 63.0186119, -85.9346924, 63.3149261, -148.8050079, 148.9533081
29: -119.1284180, 76.5195312, -119.5844574, 77.1115189, -196.2399292, 196.1039886
30: -102.5295258, 79.3458557, -103.0833740, 79.9660110, -182.4955139, 182.4292297
31: -106.1062775, 66.8025055, -106.8487015, 67.3284302, -173.4347076, 173.6512146
32: -99.8052750, 73.3462524, -100.1650848, 73.6642990, -173.4695740, 173.5113220
33: -140.3627472, 80.5459747, -140.9838409, 81.0389175, -221.4016724, 221.5298157
34: -119.5778732, 72.6952972, -120.0697250, 73.0201492, -192.5980225, 192.7650146
35: -119.9920731, 70.1504059, -120.5807800, 70.4830322, -190.4750977, 190.7311707
36: -117.2976456, 69.5986481, -117.8032990, 69.8225708, -187.1202087, 187.4019470
37: -164.3506622, 73.8736267, -164.8316040, 74.1796417, -238.5302734, 238.7052307
38: -145.1417847, 86.0872040, -145.7966614, 86.4411240, -231.5829163, 231.8838501
39: -167.7562103, 77.8291779, -168.3895111, 78.2090912, -245.9652710, 246.2186737
40: -134.9875183, 73.6614532, -135.4805603, 73.8615723, -208.8490906, 209.1420135
41: -100.4549255, 67.0501328, -100.7958679, 67.3463287, -167.8012543, 167.8460083
42: -75.5573273, 65.2062378, -75.8896179, 65.7949982, -141.3523254, 141.0958557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=679, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 647

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3440903, upper bound: 103.3588494
time: 344.35 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3440903, upper bound: 103.4300530
time: 127.24 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -125.2418594, 84.5267258, -125.0134048, 84.4634933, -209.7053375, 209.5401306
1: -70.3811569, 74.4161148, -70.2258606, 74.3771362, -144.7583008, 144.6419678
2: -63.3147202, 71.4206161, -63.0682487, 71.3798370, -134.6945496, 134.4888611
3: -72.9009705, 86.4724197, -72.6172333, 86.4170837, -159.3180542, 159.0896454
4: -75.9679413, 84.7325439, -75.6937332, 84.6831055, -160.6510315, 160.4262695
5: -68.0910034, 90.8378143, -67.8427887, 90.7877884, -158.8787842, 158.6806030
6: -102.8332291, 76.0532913, -102.7488098, 75.9586411, -178.7918701, 178.8020935
7: -83.9977264, 91.3607254, -83.7851944, 91.3206177, -175.3183441, 175.1459045
8: -89.1530151, 101.8341217, -88.9017792, 101.7811584, -190.9341431, 190.7359009
9: -78.5203476, 81.9525452, -78.3759384, 81.8684082, -160.3887634, 160.3284912
10: -111.3669128, 118.5229263, -111.2455673, 118.2066879, -229.5735931, 229.7684937
11: -111.0840225, 84.3518677, -110.9939880, 83.9069214, -194.9909363, 195.3458099
12: -111.3998108, 89.7900314, -111.3324585, 89.3760147, -200.7758179, 201.1224670
13: -110.6904373, 100.6671448, -110.4251938, 100.5430145, -211.2334595, 211.0923462
14: -163.2244263, 84.4407806, -163.0815277, 84.0976791, -247.3221130, 247.5222931
15: -92.0351715, 81.7765045, -91.8728333, 81.6910934, -173.7262573, 173.6493378
16: -118.4884338, 97.8474579, -118.3395462, 97.6870117, -216.1754456, 216.1870117
17: -164.6698914, 120.4564667, -164.5534668, 119.9214859, -284.5913696, 285.0099487
18: -101.9963760, 85.3280487, -101.8796844, 84.9677734, -186.9641418, 187.2077332
19: -85.3433380, 47.9701157, -85.2698746, 47.7573128, -133.1006470, 133.2399902
20: -74.9259644, 57.8253822, -74.8491287, 57.6457977, -132.5717621, 132.6745148
21: -104.7725906, 63.7503967, -104.6940308, 63.4281998, -168.2007904, 168.4444275
22: -113.3680115, 73.4715195, -113.3125000, 73.1801605, -186.5481567, 186.7839661
23: -86.5876465, 58.8251953, -86.5215149, 58.6216125, -145.2092438, 145.3466949
24: -103.7151718, 69.5845947, -103.6430664, 69.4028778, -173.1180420, 173.2276611
25: -91.0810547, 68.4036255, -91.0175095, 68.2227859, -159.3038330, 159.4211426
26: -122.4286728, 90.4155197, -122.3437805, 89.9897003, -212.4183655, 212.7593079
27: -104.6115875, 74.4076462, -104.5020065, 74.1672974, -178.7788696, 178.9096527
28: -85.7802582, 63.3812103, -85.7081985, 63.2053146, -148.9855652, 149.0894012
29: -119.4451294, 77.3056641, -119.3793488, 76.9188690, -196.3639984, 196.6850128
30: -102.9111023, 80.1224976, -102.8384476, 79.8159714, -182.7270813, 182.9609375
31: -106.6347885, 67.5034180, -106.5345306, 67.2862930, -173.9210815, 174.0379333
32: -100.1343689, 73.7098007, -100.0509033, 73.5372467, -173.6716003, 173.7606964
33: -141.1343536, 80.9134216, -140.8703918, 80.8306122, -221.9649506, 221.7838135
34: -120.1530762, 73.0073395, -119.9928665, 72.9032288, -193.0563049, 193.0002136
35: -120.7290115, 70.4305115, -120.5446625, 70.3706970, -191.0996857, 190.9751587
36: -117.8998795, 69.8174133, -117.7906723, 69.7450256, -187.6448975, 187.6080933
37: -164.8117065, 74.2181931, -164.7039185, 74.0803528, -238.8920593, 238.9221191
38: -145.8846741, 86.4366608, -145.6839905, 86.3503265, -232.2349854, 232.1206360
39: -168.5223999, 78.0943146, -168.2931519, 78.0366440, -246.5590515, 246.3874512
40: -135.5607147, 73.8494568, -135.4219208, 73.8119431, -209.3726501, 209.2713623
41: -100.7888794, 67.3868561, -100.6949310, 67.2814255, -168.0702820, 168.0817871
42: -75.8311310, 65.9028778, -75.7596359, 65.6725769, -141.5036926, 141.6625061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.2901960, upper bound: 103.3519709
time: 88.88 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.4476803, upper bound: 103.3767771
time: 324.49 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -125.2643814, 84.5312500, -125.3509750, 84.6473083, -209.9116821, 209.8822327
1: -70.3963470, 74.4194412, -70.4441986, 74.5226669, -144.9190063, 144.8636475
2: -63.3402901, 71.4238281, -63.3727036, 71.6896439, -135.0299377, 134.7965393
3: -72.9312134, 86.4759674, -72.9739075, 86.7983475, -159.7295532, 159.4498596
4: -75.9960556, 84.7362137, -76.0388336, 84.9647675, -160.9608002, 160.7750549
5: -68.1177979, 90.8414612, -68.1614761, 91.1805725, -159.2983704, 159.0029297
6: -102.8392639, 76.0370483, -102.9468155, 76.1022568, -178.9414825, 178.9838562
7: -84.0163422, 91.3634796, -84.0760498, 91.4896240, -175.5059509, 175.4395294
8: -89.1788025, 101.8376617, -89.2216492, 102.0837784, -191.2625732, 191.0593109
9: -78.5370331, 81.9549408, -78.6054535, 82.0445938, -160.5816345, 160.5603943
10: -111.3776474, 118.5570755, -111.5640411, 118.6839981, -230.0616455, 230.1210938
11: -111.0873032, 84.4006500, -111.4983978, 84.4395065, -195.5268097, 195.8990479
12: -111.4049377, 89.8336487, -111.8553772, 89.8965912, -201.3015289, 201.6890259
13: -110.7015686, 100.6747284, -110.7170486, 100.9294510, -211.6310120, 211.3917542
14: -163.2368469, 84.4784546, -163.5176392, 84.5146790, -247.7515259, 247.9960938
15: -92.0192719, 81.7807770, -92.0995331, 81.8912811, -173.9105530, 173.8803101
16: -118.4993286, 97.8193283, -118.6657104, 97.8605804, -216.3599091, 216.4850464
17: -164.6800537, 120.5116425, -165.0823212, 120.5534821, -285.2335205, 285.5939636
18: -102.0046921, 85.3656540, -102.3213043, 85.4079742, -187.4126587, 187.6869507
19: -85.3489838, 47.9934425, -85.6359863, 48.0174141, -133.3663940, 133.6294250
20: -74.9311829, 57.8447762, -75.1285095, 57.8702583, -132.8014221, 132.9732666
21: -104.7775803, 63.7870064, -105.1516190, 63.8146248, -168.5921936, 168.9386139
22: -113.3727951, 73.4993896, -113.5421677, 73.5335464, -186.9063263, 187.0415649
23: -86.5936890, 58.8473892, -86.8232117, 58.8833694, -145.4770355, 145.6705933
24: -103.7198715, 69.6049957, -103.9000168, 69.6288910, -173.3487396, 173.5050049
25: -91.0843964, 68.4198532, -91.1842651, 68.4428406, -159.5272369, 159.6041260
26: -122.4334183, 90.4615784, -122.8793869, 90.5150070, -212.9484253, 213.3409424
27: -104.6190872, 74.4337845, -104.8320770, 74.4530945, -179.0721741, 179.2658539
28: -85.7870483, 63.3995247, -86.0044098, 63.4296036, -149.2166443, 149.4039307
29: -119.4491577, 77.3458939, -119.6581650, 77.3663635, -196.8155212, 197.0040588
30: -102.9140778, 80.1554565, -103.1598053, 80.2101440, -183.1242218, 183.3152466
31: -106.6431732, 67.5269165, -106.9491196, 67.5540009, -174.1971436, 174.4760437
32: -100.1405487, 73.7275620, -100.2580338, 73.7789307, -173.9194794, 173.9855804
33: -141.1616974, 80.9191284, -141.2227325, 81.1080093, -222.2696991, 222.1418457
34: -120.1696396, 73.0152969, -120.2440414, 73.1030273, -193.2726746, 193.2593384
35: -120.7458191, 70.4348373, -120.8087082, 70.5402527, -191.2860718, 191.2435455
36: -117.9075928, 69.8225403, -117.9892731, 69.8784637, -187.7860565, 187.8118134
37: -164.8199768, 74.2208557, -164.9665680, 74.2781067, -239.0980835, 239.1874237
38: -145.9041290, 86.4377747, -146.0258789, 86.5140228, -232.4181213, 232.4636536
39: -168.5441284, 78.0985870, -168.6274414, 78.2632828, -246.8074036, 246.7259979
40: -135.5707092, 73.8317108, -135.6535797, 73.9055252, -209.4762268, 209.4852600
41: -100.7967300, 67.3743744, -100.8929443, 67.4400482, -168.2367859, 168.2673035
42: -75.8365555, 65.9253159, -75.9625244, 66.0096893, -141.8462524, 141.8878326

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.2905036, upper bound: 103.4248440
time: 100.50 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.4476803, upper bound: 103.4476804
time: 102.39 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 205.09 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 205.09
Output dim: 5, lower bound: -103.3440903, upper bound: 103.2849830
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 205.09
Output dim: 5, lower bound: -103.3440903, upper bound: 103.3578408
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 205.09
Output dim: 5, lower bound: -103.3440903, upper bound: 103.3588494
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 205.09
Output dim: 5, lower bound: -103.3440903, upper bound: 103.4300530
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 205.09
Output dim: 5, lower bound: -103.2901960, upper bound: 103.3519709
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 205.09
Output dim: 5, lower bound: -103.4476803, upper bound: 103.3767771
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 205.09
Output dim: 5, lower bound: -103.2905036, upper bound: 103.4248440
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 205.09
Output dim: 5, lower bound: -103.4476803, upper bound: 103.4476804

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -124.3106537, 83.9432220, -124.7882004, 84.3107452, -208.6213989, 208.7314148
1: -69.7218628, 73.9906311, -70.0684357, 74.2724304, -143.9942627, 144.0590515
2: -62.4784164, 70.8907776, -62.8613014, 71.2654266, -133.7438202, 133.7520752
3: -71.8663177, 85.7205048, -72.3631821, 86.2552414, -158.1215515, 158.0836792
4: -75.0365982, 84.3248749, -75.4476776, 84.6001816, -159.6367798, 159.7725525
5: -67.1804504, 90.0713501, -67.6346130, 90.6214294, -157.8018646, 157.7059631
6: -102.3814545, 75.3168488, -102.6337051, 75.7514496, -178.1329041, 177.9505310
7: -83.0748138, 90.6694946, -83.5915680, 91.1367645, -174.2115784, 174.2610626
8: -88.2527924, 101.2224274, -88.6785965, 101.6464386, -189.8992310, 189.9010010
9: -77.9058228, 81.4106903, -78.2078400, 81.7501984, -159.6559906, 159.6185150
10: -110.3395081, 116.8569489, -111.0366669, 117.7845535, -228.1240540, 227.8936157
11: -110.3697281, 82.8916931, -110.8628082, 83.5017700, -193.8714600, 193.7545013
12: -110.2475891, 88.1771393, -111.0734253, 89.0082932, -199.2558594, 199.2505646
13: -109.4561539, 99.8313751, -110.0787430, 100.3809357, -209.8370667, 209.9101105
14: -161.9625549, 83.0981674, -162.7900085, 83.7757568, -245.7382812, 245.8881836
15: -90.9073410, 81.2812881, -91.5670624, 81.5834351, -172.4907837, 172.8483582
16: -117.7407608, 96.8313751, -118.1553802, 97.4054413, -215.1462097, 214.9867401
17: -163.6412964, 118.6509247, -164.3369446, 119.4508133, -283.0921021, 282.9878540
18: -101.2251282, 84.1947784, -101.7070084, 84.6441345, -185.8692627, 185.9017944
19: -84.8080292, 47.3545074, -85.1702652, 47.5763626, -132.3843689, 132.5247650
20: -74.3675690, 57.2684097, -74.7321930, 57.4907227, -131.8582764, 132.0005951
21: -104.1410675, 62.8238716, -104.5760727, 63.1663895, -167.3074341, 167.3999481
22: -112.5008163, 72.4953766, -113.0828705, 72.9555283, -185.4563446, 185.5782471
23: -86.1044998, 58.1047935, -86.4273987, 58.4160919, -144.5205994, 144.5321960
24: -103.1783218, 69.0164948, -103.5297852, 69.2378693, -172.4161835, 172.5462646
25: -90.5590210, 67.7293396, -90.8930511, 68.0473633, -158.6063843, 158.6223907
26: -121.1713791, 88.9315186, -122.0567398, 89.6479111, -210.8192902, 210.9882355
27: -103.9753647, 73.7722168, -104.3558044, 73.9841461, -177.9594879, 178.1280212
28: -85.3461914, 62.8982582, -85.6142731, 63.0700340, -148.4162292, 148.5125275
29: -118.7433167, 76.1615906, -119.2060699, 76.6437302, -195.3870544, 195.3676453
30: -102.3694382, 79.1011887, -102.7322540, 79.5289993, -181.8984375, 181.8334351
31: -105.9313736, 66.6660004, -106.3982162, 67.0422897, -172.9736481, 173.0642090
32: -99.6038055, 73.1652679, -99.9124832, 73.4005051, -173.0043030, 173.0777588
33: -140.1118927, 80.4059219, -140.6052551, 80.7292862, -220.8411407, 221.0111694
34: -119.3210754, 72.5056763, -119.7906876, 72.7770386, -192.0981140, 192.2963562
35: -119.7760620, 70.0040359, -120.2944641, 70.2792358, -190.0552979, 190.2984924
36: -117.1073074, 69.4906311, -117.5714035, 69.6650848, -186.7723846, 187.0620422
37: -164.1014099, 73.7678986, -164.5219116, 73.9627533, -238.0641632, 238.2898102
38: -144.9018250, 85.9550247, -145.4292145, 86.2476959, -231.1495056, 231.3842163
39: -167.4983673, 77.6990051, -168.0129395, 77.9539719, -245.4523315, 245.7119446
40: -134.7610321, 73.4373016, -135.2220154, 73.7044678, -208.4654999, 208.6593018
41: -100.2991486, 66.8034973, -100.5758896, 67.1229248, -167.4220734, 167.3793945
42: -75.4721909, 64.9712524, -75.6646423, 65.4275894, -140.8997803, 140.6358948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=678, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3125181, upper bound: 103.1820956
time: 131.76 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3125181, upper bound: 103.2730458
time: 129.40 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -124.5428772, 84.2591019, -124.8166122, 84.3968964, -208.9397583, 209.0757141
1: -69.8914642, 74.2056961, -70.0855713, 74.3306274, -144.2220917, 144.2912598
2: -62.6733665, 71.1371002, -62.8779564, 71.3346939, -134.0080566, 134.0150604
3: -72.1052017, 86.0652084, -72.3791504, 86.3510590, -158.4562378, 158.4443359
4: -75.1989822, 84.4305420, -75.4676514, 84.6256561, -159.8246460, 159.8981934
5: -67.4479675, 90.4568939, -67.6548080, 90.7283478, -158.1763153, 158.1116943
6: -102.5214386, 75.6264496, -102.6563492, 75.8368073, -178.3582458, 178.2827759
7: -83.4225845, 91.1376877, -83.6223907, 91.2707825, -174.6933594, 174.7600708
8: -88.4530563, 101.4860077, -88.6933823, 101.7183228, -190.1713867, 190.1793823
9: -78.1277161, 81.6055756, -78.2666473, 81.7680893, -159.8958130, 159.8721924
10: -110.6994781, 117.1864166, -111.1322098, 117.8086166, -228.5080872, 228.3186340
11: -110.5041046, 83.0161209, -110.9041901, 83.5144958, -194.0185699, 193.9203186
12: -110.8911667, 88.6284027, -111.2583313, 89.0313492, -199.9225159, 199.8867188
13: -109.7879486, 100.0881195, -110.1587906, 100.4211197, -210.2090759, 210.2469177
14: -162.5548401, 83.4227982, -162.9449158, 83.7894821, -246.3442841, 246.3677063
15: -91.3358994, 81.5151215, -91.6786652, 81.6099777, -172.9458771, 173.1937866
16: -117.9850693, 97.0548706, -118.1973572, 97.4593201, -215.4443970, 215.2522278
17: -164.0794067, 118.9708099, -164.4539490, 119.4712830, -283.5506897, 283.4247437
18: -101.4505234, 84.3076706, -101.7597122, 84.6603546, -186.1108704, 186.0673828
19: -84.9177094, 47.4057541, -85.1988449, 47.5852890, -132.5029907, 132.6045990
20: -74.5374374, 57.3472519, -74.7699509, 57.5025330, -132.0399780, 132.1171875
21: -104.2706757, 62.9244614, -104.6154633, 63.1800690, -167.4507446, 167.5399170
22: -112.9976425, 72.7976990, -113.2226181, 72.9803467, -185.9779816, 186.0203094
23: -86.2295532, 58.1789093, -86.4559174, 58.4333229, -144.6628723, 144.6348267
24: -103.3246002, 69.0884628, -103.5581055, 69.2529068, -172.5774689, 172.6465607
25: -90.7726059, 67.8975906, -90.9493790, 68.0706940, -158.8432922, 158.8469696
26: -121.8714752, 89.3237381, -122.2473373, 89.6679840, -211.5394592, 211.5710754
27: -104.1972351, 73.9092255, -104.3852997, 74.0192413, -178.2164764, 178.2945251
28: -85.4763184, 62.9886513, -85.6378632, 63.0907402, -148.5670471, 148.6265106
29: -119.0969849, 76.4633560, -119.3032303, 76.6657562, -195.7627411, 195.7665863
30: -102.5097961, 79.2707062, -102.7610626, 79.5695801, -182.0793762, 182.0317688
31: -106.0841827, 66.7374115, -106.4319229, 67.0583191, -173.1424866, 173.1693420
32: -99.7830048, 73.3152542, -99.9563141, 73.4221497, -173.2051544, 173.2715607
33: -140.3208923, 80.5270996, -140.6311340, 80.7608719, -221.0817108, 221.1582336
34: -119.5507965, 72.6703796, -119.8183823, 72.8205490, -192.3713379, 192.4887695
35: -119.9615173, 70.1345596, -120.3176193, 70.3135376, -190.2750549, 190.4521790
36: -117.2610474, 69.5839386, -117.6034012, 69.6904144, -186.9514465, 187.1873474
37: -164.3134155, 73.8624954, -164.5667419, 73.9812469, -238.2946625, 238.4292297
38: -145.1100769, 86.0751495, -145.4559784, 86.2774582, -231.3874969, 231.5311279
39: -167.6833801, 77.8158112, -168.0523071, 77.9819565, -245.6653442, 245.8680878
40: -134.9567261, 73.6658325, -135.2483978, 73.7702866, -208.7269897, 208.9142303
41: -100.4349747, 67.0280151, -100.5971756, 67.1844025, -167.6193848, 167.6251831
42: -75.5425110, 65.1331558, -75.6860809, 65.4537811, -140.9962921, 140.8192291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=678, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3125181, upper bound: 103.2653070
time: 129.57 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3125181, upper bound: 103.3483513
time: 105.27 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -124.3322601, 83.9471054, -125.1221008, 84.4947662, -208.8270111, 209.0691986
1: -69.7356262, 73.9938965, -70.2836914, 74.4174652, -144.1530762, 144.2775574
2: -62.5021706, 70.8937836, -63.1636505, 71.5748138, -134.0769653, 134.0574341
3: -71.8947449, 85.7241058, -72.7178574, 86.6359100, -158.5306396, 158.4419556
4: -75.0632935, 84.3279572, -75.7904434, 84.8807526, -159.9440460, 160.1183929
5: -67.2059174, 90.0747528, -67.9506226, 91.0138092, -158.2197266, 158.0253754
6: -102.3867493, 75.2973328, -102.8272018, 75.8933640, -178.2800903, 178.1245422
7: -83.0916443, 90.6722641, -83.8752975, 91.3052750, -174.3969116, 174.5475616
8: -88.2767029, 101.2258148, -88.9962006, 101.9479599, -190.2246552, 190.2220001
9: -77.9196396, 81.4126587, -78.4363098, 81.9246216, -159.8442688, 159.8489532
10: -110.3489456, 116.8885345, -111.3534012, 118.2582550, -228.6072083, 228.2419434
11: -110.3718414, 82.9387207, -111.3656464, 84.0316162, -194.4034576, 194.3043518
12: -110.2523193, 88.2194214, -111.5959015, 89.5266113, -199.7789307, 199.8153229
13: -109.4645538, 99.8389969, -110.3650360, 100.7627792, -210.2273254, 210.2040405
14: -161.9743958, 83.1344376, -163.2250977, 84.1907043, -246.1650848, 246.3595276
15: -90.8815460, 81.2849426, -91.7876587, 81.7787628, -172.6603088, 173.0725861
16: -117.7505569, 96.8022232, -118.4761581, 97.5753632, -215.3259277, 215.2783813
17: -163.6510925, 118.7032471, -164.8651123, 120.0799942, -283.7310791, 283.5683594
18: -101.2332611, 84.2297668, -102.1468964, 85.0821228, -186.3153839, 186.3766632
19: -84.8134537, 47.3759460, -85.5348663, 47.8343811, -132.6478271, 132.9108124
20: -74.3724213, 57.2865372, -75.0103149, 57.7138252, -132.0862427, 132.2968445
21: -104.1456833, 62.8588638, -105.0316162, 63.5509033, -167.6965790, 167.8904724
22: -112.5054169, 72.5204468, -113.3117065, 73.3002167, -185.8056335, 185.8321533
23: -86.1103210, 58.1256332, -86.7282791, 58.6761627, -144.7864685, 144.8539124
24: -103.1830292, 69.0355072, -103.7857590, 69.4628448, -172.6458740, 172.8212585
25: -90.5624847, 67.7435379, -91.0588913, 68.2632294, -158.8257141, 158.8024139
26: -121.1761932, 88.9756470, -122.5915527, 90.1692581, -211.3454285, 211.5671844
27: -103.9826508, 73.7968674, -104.6839447, 74.2685547, -178.2512054, 178.4808044
28: -85.3527145, 62.9148178, -85.9099274, 63.2924347, -148.6451416, 148.8247375
29: -118.7471848, 76.1997528, -119.4839554, 77.0852814, -195.8324585, 195.6836853
30: -102.3719254, 79.1324463, -103.0517654, 79.9208450, -182.2927551, 182.1842041
31: -105.9398117, 66.6873093, -106.8119431, 67.3075867, -173.2473907, 173.4992523
32: -99.6091766, 73.1824036, -100.1184692, 73.6409760, -173.2501221, 173.3008575
33: -140.1384583, 80.4110641, -140.9557953, 81.0043030, -221.1427612, 221.3668518
34: -119.3372650, 72.5120239, -120.0403442, 72.9744263, -192.3116760, 192.5523376
35: -119.7915344, 70.0077820, -120.5557556, 70.4461212, -190.2376404, 190.5635376
36: -117.1144180, 69.4951324, -117.7680664, 69.7943268, -186.9087372, 187.2631989
37: -164.1092224, 73.7696228, -164.7826996, 74.1598740, -238.2691040, 238.5523224
38: -144.9197693, 85.9556274, -145.7680054, 86.4096375, -231.3294067, 231.7236328
39: -167.5189209, 77.7030029, -168.3444214, 78.1792374, -245.6981201, 246.0474243
40: -134.7705536, 73.4171982, -135.4498901, 73.7943039, -208.5648499, 208.8670959
41: -100.3060303, 66.7927780, -100.7716599, 67.2811356, -167.5871582, 167.5644379
42: -75.4763184, 64.9951630, -75.8654480, 65.7634888, -141.2398071, 140.8605957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=678, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3125181, upper bound: 103.2693330
time: 149.70 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3125181, upper bound: 103.2693330
time: 133.69 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -124.5645142, 84.2630081, -125.1517944, 84.5813446, -209.1458435, 209.4147949
1: -69.9052734, 74.2089310, -70.3016205, 74.4760132, -144.3812866, 144.5105591
2: -62.6971817, 71.1400604, -63.1804581, 71.6443634, -134.3415527, 134.3205261
3: -72.1336594, 86.0687103, -72.7340317, 86.7321625, -158.8658142, 158.8027344
4: -75.2257156, 84.4336548, -75.8106308, 84.9070587, -160.1327515, 160.2442932
5: -67.4734726, 90.4602203, -67.9713974, 91.1209641, -158.5944214, 158.4316101
6: -102.5268097, 75.6071930, -102.8520355, 75.9795151, -178.5063171, 178.4592285
7: -83.4395599, 91.1403046, -83.9086533, 91.4395752, -174.8791351, 175.0489502
8: -88.4770203, 101.4893570, -89.0111847, 102.0204620, -190.4974823, 190.5005341
9: -78.1397781, 81.6076355, -78.4915466, 81.9430084, -160.0827942, 160.0991821
10: -110.7088623, 117.2180862, -111.4504166, 118.2825546, -228.9914093, 228.6685028
11: -110.5066299, 83.0630264, -111.4087982, 84.0446625, -194.5513000, 194.4718170
12: -110.8957977, 88.6707382, -111.7812576, 89.5499115, -200.4457092, 200.4519958
13: -109.7890244, 100.0958939, -110.4383316, 100.8054047, -210.5944214, 210.5342255
14: -162.5665894, 83.4591522, -163.3806915, 84.2051544, -246.7717438, 246.8398438
15: -91.3104706, 81.5188446, -91.9000854, 81.8078766, -173.1183472, 173.4189301
16: -117.9950485, 97.0256805, -118.5206757, 97.6297913, -215.6248322, 215.5463562
17: -164.0891113, 119.0232086, -164.9825134, 120.1010132, -284.1901245, 284.0057373
18: -101.4588318, 84.3426514, -102.2026138, 85.0987167, -186.5575409, 186.5452576
19: -84.9230652, 47.4271927, -85.5641785, 47.8435860, -132.7666473, 132.9913635
20: -74.5423737, 57.3653717, -75.0488510, 57.7258911, -132.2682495, 132.4142151
21: -104.2752991, 62.9594803, -105.0721283, 63.5647888, -167.8400879, 168.0316010
22: -113.0020752, 72.8229675, -113.4519958, 73.3282928, -186.3303375, 186.2749634
23: -86.2353668, 58.1997070, -86.7573395, 58.6936455, -144.9290161, 144.9570465
24: -103.3294067, 69.1074677, -103.8149033, 69.4782410, -172.8076477, 172.9223633
25: -90.7760620, 67.9118958, -91.1155548, 68.2881165, -159.0641785, 159.0274506
26: -121.8760986, 89.3678284, -122.7827148, 90.1904297, -212.0665283, 212.1505127
27: -104.2046967, 73.9337769, -104.7147064, 74.3039856, -178.5086823, 178.6484833
28: -85.4828491, 63.0051765, -85.9339447, 63.3134842, -148.7963257, 148.9391174
29: -119.1007843, 76.5017014, -119.5816803, 77.1093597, -196.2101288, 196.0833740
30: -102.5124283, 79.3018112, -103.0815353, 79.9617004, -182.4741211, 182.3833466
31: -106.0928040, 66.7587280, -106.8470993, 67.3241348, -173.4169312, 173.6058350
32: -99.7883987, 73.3323898, -100.1632385, 73.6628571, -173.4512634, 173.4956207
33: -140.3475037, 80.5322037, -140.9821777, 81.0373535, -221.3848572, 221.5143738
34: -119.5670624, 72.6767044, -120.0685501, 73.0182266, -192.5852661, 192.7452545
35: -119.9769440, 70.1382141, -120.5791550, 70.4815979, -190.4585266, 190.7173767
36: -117.2681961, 69.5884705, -117.8003845, 69.8212280, -187.0894165, 187.3888550
37: -164.3212433, 73.8635178, -164.8285217, 74.1785355, -238.4997711, 238.6920471
38: -145.1280212, 86.0757828, -145.7951965, 86.4398804, -231.5678558, 231.8709717
39: -167.7039185, 77.8198624, -168.3843384, 78.2080994, -245.9119720, 246.2041931
40: -134.9664001, 73.6457977, -135.4781647, 73.8599854, -208.8263855, 209.1239624
41: -100.4419479, 67.0180283, -100.7943497, 67.3432159, -167.7851410, 167.8123779
42: -75.5467377, 65.1570663, -75.8883438, 65.7901306, -141.3368683, 141.0454102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=679, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3125181, upper bound: 103.3489556
time: 93.68 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3125181, upper bound: 103.4210167
time: 89.07 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -124.5972290, 84.2949524, -124.8312225, 84.4190674, -209.0162964, 209.1261749
1: -69.9339066, 74.2610321, -70.0982056, 74.3450012, -144.2789001, 144.3592377
2: -62.6483192, 71.1920166, -62.8744240, 71.3497772, -133.9980774, 134.0664368
3: -72.1535034, 86.1631622, -72.3984833, 86.3673096, -158.5208130, 158.5616455
4: -75.2163849, 84.4922409, -75.4768066, 84.6399689, -159.8563538, 159.9690247
5: -67.4058838, 90.5278778, -67.6449509, 90.7422791, -158.1481628, 158.1728210
6: -102.5414734, 75.5839233, -102.6713104, 75.8282318, -178.3696899, 178.2552338
7: -83.3685532, 91.1682510, -83.6089020, 91.2830658, -174.6516113, 174.7771606
8: -88.4047089, 101.5421295, -88.6843719, 101.7386169, -190.1433258, 190.2264862
9: -78.2473602, 81.4478455, -78.3166885, 81.7264099, -159.9737549, 159.7645264
10: -110.8252792, 117.2422028, -111.1588287, 117.8312378, -228.6565247, 228.4010315
11: -110.7319641, 83.3597412, -110.9219437, 83.6110535, -194.3430176, 194.2816772
12: -110.9495392, 88.4505310, -111.2833710, 88.9841309, -199.9336700, 199.7338867
13: -110.3195572, 100.2628479, -110.3238754, 100.4327164, -210.7522736, 210.5867004
14: -162.7082520, 83.6593704, -162.9748840, 83.8709488, -246.5791931, 246.6342468
15: -91.4038467, 81.4971466, -91.6996002, 81.6183701, -173.0222168, 173.1967468
16: -118.0817261, 97.0744781, -118.2381287, 97.4671631, -215.5488739, 215.3126068
17: -164.2699738, 119.3046417, -164.4835968, 119.5860138, -283.8559875, 283.7882080
18: -101.6081161, 84.7180557, -101.7928925, 84.7925034, -186.4006195, 186.5109558
19: -85.0379639, 47.6228218, -85.2117386, 47.6558151, -132.6937866, 132.8345642
20: -74.6350861, 57.5169983, -74.7823715, 57.5546989, -132.1897736, 132.2993469
21: -104.4074936, 63.2025032, -104.6308670, 63.2680931, -167.6755829, 167.8333740
22: -113.0302887, 72.9612274, -113.2267151, 73.0339737, -186.0642395, 186.1879425
23: -86.3457947, 58.4952621, -86.4660950, 58.5260811, -144.8718567, 144.9613495
24: -103.4114532, 69.4680405, -103.5628357, 69.3714447, -172.7828827, 173.0308533
25: -90.8711700, 68.1010132, -90.9662323, 68.1393051, -159.0104675, 159.0672455
26: -122.0167084, 89.4738770, -122.2677307, 89.7180405, -211.7347412, 211.7416077
27: -104.1622238, 74.2736359, -104.3771133, 74.1322021, -178.2944336, 178.6507568
28: -85.5491180, 63.2079010, -85.6495667, 63.1589508, -148.7080536, 148.8574524
29: -119.1834259, 76.6818008, -119.3153839, 76.7378693, -195.9212799, 195.9971924
30: -102.6789627, 79.6616364, -102.7780533, 79.6857224, -182.3646851, 182.4396667
31: -106.2592773, 67.0963745, -106.4478455, 67.1678925, -173.4271698, 173.5441895
32: -99.8237000, 73.2611313, -99.9865341, 73.4071732, -173.2308655, 173.2476654
33: -140.4977112, 80.6092529, -140.6881409, 80.7686462, -221.2663574, 221.2973785
34: -119.6574478, 72.7488098, -119.8517990, 72.8428802, -192.5003357, 192.6006012
35: -120.1158981, 70.1788788, -120.3657227, 70.3251572, -190.4410400, 190.5445862
36: -117.5058060, 69.6229019, -117.6794357, 69.6989059, -187.2047119, 187.3023376
37: -164.4374695, 73.8896027, -164.6064148, 73.9922485, -238.4296875, 238.4960175
38: -145.3014679, 86.2100220, -145.5201721, 86.2985992, -231.6000366, 231.7301636
39: -168.0130157, 77.8792648, -168.1555786, 77.9874496, -246.0004578, 246.0348358
40: -135.0944977, 73.6691055, -135.2951050, 73.7703094, -208.8647766, 208.9641876
41: -100.5158997, 67.0381470, -100.6205444, 67.1858368, -167.7017365, 167.6586761
42: -75.5811005, 65.1648712, -75.7052460, 65.4577179, -141.0388184, 140.8701019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=679, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 647

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.2004955, upper bound: 103.3395297
time: 102.96 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.2004955, upper bound: 103.3395297
time: 112.07 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -125.2142334, 84.5200958, -125.0104828, 84.4628143, -209.6770477, 209.5305786
1: -70.3617172, 74.4109879, -70.2237549, 74.3766174, -144.7383423, 144.6347351
2: -63.2904701, 71.4151001, -63.0656281, 71.3792725, -134.6697388, 134.4807129
3: -72.8709030, 86.4630127, -72.6140976, 86.4161224, -159.2870178, 159.0771179
4: -75.9409485, 84.7233124, -75.6907806, 84.6821594, -160.6230927, 160.4140930
5: -68.0635681, 90.8291931, -67.8399200, 90.7868958, -158.8504639, 158.6691132
6: -102.8179474, 75.9955750, -102.7472839, 75.9529495, -178.7708893, 178.7428284
7: -83.9709167, 91.3549042, -83.7824326, 91.3199921, -175.2909088, 175.1373291
8: -89.1253357, 101.8266144, -88.8988113, 101.7803497, -190.9056854, 190.7254181
9: -78.5105438, 81.9333344, -78.3749161, 81.8663940, -160.3769379, 160.3082581
10: -111.3518448, 118.4742432, -111.2440186, 118.2013931, -229.5532227, 229.7182312
11: -111.0696869, 84.3161087, -110.9925232, 83.9030914, -194.9727783, 195.3086243
12: -111.3900375, 89.7434692, -111.3314438, 89.3710403, -200.7610779, 201.0748901
13: -110.6588287, 100.6482697, -110.4214096, 100.5410233, -211.1998596, 211.0696564
14: -163.2066498, 84.4130325, -163.0796204, 84.0947266, -247.3013611, 247.4926453
15: -91.9639587, 81.7617645, -91.8649368, 81.6896362, -173.6535797, 173.6267090
16: -118.4672623, 97.8053589, -118.3373871, 97.6827240, -216.1499939, 216.1427460
17: -164.6579590, 120.4166260, -164.5522308, 119.9172211, -284.5751343, 284.9688721
18: -101.9808502, 85.3031311, -101.8780518, 84.9652557, -186.9461060, 187.1811829
19: -85.3336029, 47.9555817, -85.2688675, 47.7557793, -133.0893555, 133.2244568
20: -74.9153748, 57.8138733, -74.8480377, 57.6444969, -132.5598755, 132.6618958
21: -104.7602921, 63.7305298, -104.6927490, 63.4260864, -168.1863708, 168.4232635
22: -113.3421326, 73.4468765, -113.3098450, 73.1775970, -186.5196838, 186.7567139
23: -86.5783081, 58.8067169, -86.5205307, 58.6194267, -145.1977234, 145.3272400
24: -103.6938477, 69.5770874, -103.6408386, 69.4021072, -173.0959473, 173.2179108
25: -91.0707626, 68.3898468, -91.0164490, 68.2213440, -159.2920990, 159.4062958
26: -122.4155655, 90.3783112, -122.3423920, 89.9858475, -212.4013977, 212.7207031
27: -104.5913010, 74.3982315, -104.4998169, 74.1663361, -178.7576294, 178.8980408
28: -85.7712708, 63.3687172, -85.7072449, 63.2039871, -148.9752502, 149.0759583
29: -119.4338226, 77.2785721, -119.3780518, 76.9160309, -196.3498535, 196.6566162
30: -102.8994751, 80.0848541, -102.8372498, 79.8117599, -182.7112427, 182.9221039
31: -106.6207199, 67.4859619, -106.5330811, 67.2844543, -173.9051819, 174.0190430
32: -100.1219864, 73.6955109, -100.0495911, 73.5353394, -173.6573181, 173.7451019
33: -141.1090088, 80.9030914, -140.8677673, 80.8295441, -221.9385529, 221.7708588
34: -120.1330261, 72.9970245, -119.9907990, 72.9021683, -193.0351715, 192.9878235
35: -120.7043915, 70.4223099, -120.5420837, 70.3698730, -191.0742493, 190.9643707
36: -117.8811035, 69.8087387, -117.7887192, 69.7441177, -187.6252136, 187.5974426
37: -164.7927551, 74.1979904, -164.7018890, 74.0780182, -238.8707581, 238.8998718
38: -145.8591309, 86.4271774, -145.6813049, 86.3493500, -232.2084656, 232.1084900
39: -168.4888611, 78.0856857, -168.2897034, 78.0357666, -246.5245972, 246.3753967
40: -135.5394897, 73.8228912, -135.4196777, 73.8091049, -209.3486023, 209.2425537
41: -100.7757645, 67.3491516, -100.6935883, 67.2777176, -168.0534668, 168.0427399
42: -75.8196411, 65.8738327, -75.7584686, 65.6688309, -141.4884644, 141.6322937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=679, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 647

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3521157, upper bound: 103.3579024
time: 111.53 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.4286011, upper bound: 103.3579024
time: 91.54 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -124.6193314, 84.2992401, -125.1675720, 84.6029663, -209.2222900, 209.4667969
1: -69.9487915, 74.2644196, -70.3156433, 74.4904327, -144.4392242, 144.5800629
2: -62.6735916, 71.1951752, -63.1782455, 71.6595840, -134.3331757, 134.3734131
3: -72.1834412, 86.1666489, -72.7545013, 86.7486115, -158.9320374, 158.9211426
4: -75.2441406, 84.4957886, -75.8211823, 84.9215393, -160.1656799, 160.3169556
5: -67.4323730, 90.5314789, -67.9627991, 91.1351471, -158.5675049, 158.4942627
6: -102.5472107, 75.5676575, -102.8681488, 75.9726868, -178.5198669, 178.4358063
7: -83.3865280, 91.1710052, -83.8975067, 91.4520645, -174.8385925, 175.0685120
8: -88.4301605, 101.5456085, -89.0036469, 102.0410995, -190.4712524, 190.5492554
9: -78.2638550, 81.4505463, -78.5447693, 81.9026947, -160.1665497, 159.9953156
10: -110.8357773, 117.2757950, -111.4768753, 118.3075409, -229.1433105, 228.7526703
11: -110.7347412, 83.4078369, -111.4273605, 84.1430054, -194.8777313, 194.8351898
12: -110.9546127, 88.4936447, -111.8062363, 89.5037460, -200.4583588, 200.2998810
13: -110.3232269, 100.2699890, -110.6118164, 100.8182220, -211.1414490, 210.8818054
14: -162.7205811, 83.6968460, -163.4111938, 84.2873535, -247.0079346, 247.1080322
15: -91.3874512, 81.5010605, -91.9243088, 81.8175201, -173.2049561, 173.4253540
16: -118.0922394, 97.0468903, -118.5627365, 97.6396561, -215.7319031, 215.6096191
17: -164.2800140, 119.3591537, -165.0125122, 120.2167740, -284.4967957, 284.3716431
18: -101.6160889, 84.7553482, -102.2354813, 85.2318420, -186.8479309, 186.9908295
19: -85.0434265, 47.6459770, -85.5775604, 47.9154358, -132.9588623, 133.2235413
20: -74.6401901, 57.5362091, -75.0616760, 57.7787437, -132.4189301, 132.5978851
21: -104.4122467, 63.2389107, -105.0882797, 63.6540489, -168.0662842, 168.3271942
22: -113.0348434, 72.9882812, -113.4564896, 73.3846741, -186.4195099, 186.4447632
23: -86.3518066, 58.5172195, -86.7678986, 58.7873993, -145.1392059, 145.2850952
24: -103.4159775, 69.4883728, -103.8201294, 69.5972061, -173.0131836, 173.3084717
25: -90.8747253, 68.1169281, -91.1328964, 68.3583221, -159.2330475, 159.2498169
26: -122.0214005, 89.5194397, -122.8032608, 90.2420654, -212.2634277, 212.3226929
27: -104.1694183, 74.2996521, -104.7074890, 74.4176102, -178.5870209, 179.0071411
28: -85.5558167, 63.2260361, -85.9459229, 63.3828583, -148.9386597, 149.1719666
29: -119.1873627, 76.7214966, -119.5942764, 77.1835327, -196.3708801, 196.3157654
30: -102.6817322, 79.6941833, -103.0992813, 80.0788422, -182.7605743, 182.7934570
31: -106.2674866, 67.1196594, -106.8632355, 67.4349060, -173.7023926, 173.9828949
32: -99.8295898, 73.2789917, -100.1935349, 73.6482849, -173.4778748, 173.4725342
33: -140.5247192, 80.6146927, -141.0398254, 81.0458374, -221.5705566, 221.6545105
34: -119.6738129, 72.7565613, -120.1024323, 73.0414581, -192.7152710, 192.8589935
35: -120.1320343, 70.1829910, -120.6285400, 70.4938660, -190.6258850, 190.8115234
36: -117.5133591, 69.6277390, -117.8771973, 69.8307343, -187.3440857, 187.5049286
37: -164.4455872, 73.8916779, -164.8687744, 74.1895828, -238.6351624, 238.7604523
38: -145.3204651, 86.2111053, -145.8609009, 86.4617920, -231.7822571, 232.0720062
39: -168.0342407, 77.8833618, -168.4885864, 78.2142792, -246.2485046, 246.3719482
40: -135.1041412, 73.6514740, -135.5255737, 73.8624954, -208.9666443, 209.1770172
41: -100.5235825, 67.0257874, -100.8181381, 67.3449554, -167.8685303, 167.8439178
42: -75.5863113, 65.1866760, -75.9076538, 65.7941132, -141.3804321, 141.0943146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 647

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.2769077, upper bound: 103.4116519
time: 200.89 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.2012763, upper bound: 103.4116519
time: 215.33 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 418.52 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 418.52
Output dim: 5, lower bound: -103.3125181, upper bound: 103.1820956
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 418.52
Output dim: 5, lower bound: -103.3125181, upper bound: 103.2730458
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 418.52
Output dim: 5, lower bound: -103.3125181, upper bound: 103.2653070
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 418.52
Output dim: 5, lower bound: -103.3125181, upper bound: 103.3483513
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 418.52
Output dim: 5, lower bound: -103.3125181, upper bound: 103.2693330
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 418.52
Output dim: 5, lower bound: -103.3125181, upper bound: 103.2693330
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 418.52
Output dim: 5, lower bound: -103.3125181, upper bound: 103.3489556
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 418.52
Output dim: 5, lower bound: -103.3125181, upper bound: 103.4210167
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 418.52
Output dim: 5, lower bound: -103.2004955, upper bound: 103.3395297
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 418.52
Output dim: 5, lower bound: -103.2004955, upper bound: 103.3395297
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 418.52
Output dim: 5, lower bound: -103.3521157, upper bound: 103.3579024
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 418.52
Output dim: 5, lower bound: -103.4286011, upper bound: 103.3579024
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 418.52
Output dim: 5, lower bound: -103.2769077, upper bound: 103.4116519
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 418.52
Output dim: 5, lower bound: -103.2012763, upper bound: 103.4116519
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 418.52
Output dim: 5, lower bound: -103.4476803, upper bound: 103.4476804
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=159.03338623046875
rel_dist={5: [-103.46050891932094, 103.46050895689399]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0740828, upper bound: 99.1364372
time: 83.95 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0740828, upper bound: 99.1366734
time: 116.04 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 200.11 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 200.11
Output dim: 5, lower bound: -99.0740828, upper bound: 99.1364372
IS_A2, status: Status.UNKNOWN, split count: 1, time: 200.11
Output dim: 5, lower bound: -99.0740828, upper bound: 99.1366734

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -124.6066666, 84.2873840, -124.9842072, 84.4396973, -209.0463562, 209.2715912
1: -69.9321289, 74.2248077, -70.1936646, 74.3563843, -144.2885132, 144.4184570
2: -62.7293396, 71.1545944, -63.0642319, 71.3614273, -134.0907593, 134.2188110
3: -72.1667099, 86.0914154, -72.5829468, 86.3860931, -158.5527954, 158.6743469
4: -75.2612152, 84.4494247, -75.6651306, 84.6587982, -159.9200134, 160.1145630
5: -67.5027466, 90.4818497, -67.8447266, 90.7592773, -158.2620239, 158.3265686
6: -102.5512543, 75.7055969, -102.7119064, 75.9441376, -178.4953918, 178.4175110
7: -83.4754562, 91.1614227, -83.7846069, 91.2981567, -174.7736053, 174.9460144
8: -88.5101929, 101.5078583, -88.8741150, 101.7518539, -190.2620544, 190.3819580
9: -78.1663818, 81.6335449, -78.3920593, 81.8318939, -159.9982605, 160.0256042
10: -110.7386780, 117.2646561, -111.2162247, 117.9663162, -228.7049866, 228.4808807
11: -110.5426636, 83.1887283, -110.9683838, 83.8269043, -194.3695679, 194.1571045
12: -110.9254379, 88.7152863, -111.3058167, 89.3237381, -200.2491760, 200.0210876
13: -109.8729401, 100.1326675, -110.3225708, 100.5093307, -210.3822632, 210.4552307
14: -162.6055756, 83.4861450, -163.0468445, 84.0157776, -246.6213531, 246.5329742
15: -91.4326019, 81.5452423, -91.8112488, 81.6717072, -173.1043091, 173.3564758
16: -118.0348358, 97.1803436, -118.3008652, 97.5693512, -215.6041870, 215.4812012
17: -164.1159363, 119.0831833, -164.5391846, 119.8466187, -283.9625549, 283.6223755
18: -101.4866257, 84.3787079, -101.8370972, 84.9108582, -186.3974915, 186.2158051
19: -84.9405899, 47.4515610, -85.2478943, 47.7415543, -132.6821442, 132.6994324
20: -74.5630646, 57.3862228, -74.8218384, 57.6328163, -132.1958771, 132.2080688
21: -104.2998657, 62.9968300, -104.6717834, 63.4186516, -167.7185059, 167.6685944
22: -113.0589981, 72.8634033, -113.2761688, 73.2115402, -186.2705383, 186.1395569
23: -86.2505188, 58.2469292, -86.5009613, 58.5719452, -144.8224640, 144.7478943
24: -103.3558960, 69.1258240, -103.6068726, 69.3842773, -172.7401733, 172.7326965
25: -90.8022842, 67.9390488, -90.9917374, 68.2007370, -159.0030212, 158.9307861
26: -121.9121246, 89.4086609, -122.3070297, 89.9865036, -211.8986206, 211.7156982
27: -104.2366867, 73.9641800, -104.4603653, 74.2199936, -178.4566650, 178.4245300
28: -85.4963226, 63.0298767, -85.6863708, 63.2356377, -148.7319641, 148.7162476
29: -119.1377106, 76.5448532, -119.3533173, 76.9793625, -196.1170654, 195.8981628
30: -102.5404358, 79.3659515, -102.8109589, 79.7964783, -182.3369141, 182.1768951
31: -106.1160583, 66.8169556, -106.4981537, 67.1937561, -173.3098145, 173.3151093
32: -99.8150787, 73.3589783, -100.0147781, 73.5667648, -173.3818359, 173.3737488
33: -140.3820343, 80.5549545, -140.8110199, 80.8234406, -221.2054749, 221.3659668
34: -119.5900650, 72.7058487, -119.9133148, 72.9055023, -192.4955750, 192.6191711
35: -120.0107956, 70.1563034, -120.4121552, 70.3547363, -190.3655396, 190.5684509
36: -117.3122711, 69.6056137, -117.6399155, 69.7472458, -187.0595093, 187.2455139
37: -164.3631897, 73.8990097, -164.6344299, 74.1094742, -238.4726562, 238.5334473
38: -145.1600647, 86.1004486, -145.5741272, 86.3414154, -231.5014801, 231.6745605
39: -167.7764587, 77.8355713, -168.2000122, 78.0229950, -245.7994232, 246.0355835
40: -135.0021057, 73.6946869, -135.3245544, 73.8364029, -208.8385010, 209.0192413
41: -100.4634705, 67.0853043, -100.6594009, 67.3005066, -167.7639771, 167.7447052
42: -75.5640106, 65.2241974, -75.7351074, 65.6179504, -141.1819458, 140.9592896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0705068, upper bound: 99.0717147
time: 148.44 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0705068, upper bound: 99.1340867
time: 101.37 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -125.2836914, 84.5387802, -125.3079758, 84.5468369, -209.8305359, 209.8467407
1: -70.4095001, 74.4239502, -70.4251099, 74.4311066, -144.8406067, 144.8490601
2: -63.3581696, 71.4284592, -63.3775024, 71.4343414, -134.7925110, 134.8059692
3: -72.9513702, 86.4834442, -72.9756470, 86.4929047, -159.4442749, 159.4590759
4: -76.0163956, 84.7425385, -76.0383835, 84.7516479, -160.7680359, 160.7809143
5: -68.1348724, 90.8475342, -68.1551514, 90.8549805, -158.9898529, 159.0026855
6: -102.8498077, 76.0811539, -102.8636322, 76.1302338, -178.9800415, 178.9447937
7: -84.0359802, 91.3690567, -84.0553970, 91.3773804, -175.4133453, 175.4244537
8: -89.1975403, 101.8446808, -89.2179413, 101.8538208, -191.0513458, 191.0626221
9: -78.5486755, 81.9712906, -78.5699768, 81.9933472, -160.5420227, 160.5412598
10: -111.3890762, 118.5826645, -111.4016418, 118.6231689, -230.0122375, 229.9843140
11: -111.1012955, 84.4298019, -111.1135712, 84.4630966, -195.5643921, 195.5433655
12: -111.4130249, 89.8627014, -111.4239655, 89.8944855, -201.3075104, 201.2866669
13: -110.7405396, 100.6908951, -110.7562943, 100.7087631, -211.4492798, 211.4471741
14: -163.2510681, 84.5008774, -163.2681885, 84.5276031, -247.7786713, 247.7690735
15: -92.0728302, 81.7929382, -92.1189270, 81.8053284, -173.8781586, 173.9118500
16: -118.5161896, 97.8834991, -118.5337753, 97.9218597, -216.4380493, 216.4172668
17: -164.6907959, 120.5500946, -164.7014771, 120.5907211, -285.2815247, 285.2515869
18: -102.0187836, 85.3914032, -102.0357132, 85.4200058, -187.4387665, 187.4271240
19: -85.3569641, 48.0078011, -85.3654175, 48.0251808, -133.3821411, 133.3732147
20: -74.9406281, 57.8567352, -74.9511642, 57.8696594, -132.8102875, 132.8078918
21: -104.7878342, 63.8062553, -104.7995071, 63.8283501, -168.6161804, 168.6057587
22: -113.3832321, 73.5227127, -113.4111710, 73.5445251, -186.9277344, 186.9338684
23: -86.5999985, 58.8614655, -86.6080627, 58.8792534, -145.4792480, 145.4695129
24: -103.7299271, 69.6165771, -103.7437744, 69.6315002, -173.3614197, 173.3603516
25: -91.0931015, 68.4360886, -91.1011963, 68.4524689, -159.5455627, 159.5372925
26: -122.4458618, 90.4897308, -122.4617004, 90.5205612, -212.9664307, 212.9514313
27: -104.6329498, 74.4486694, -104.6499023, 74.4625702, -179.0955200, 179.0985718
28: -85.7934875, 63.4117355, -85.8014221, 63.4232330, -149.2167206, 149.2131500
29: -119.4589386, 77.3727341, -119.4731903, 77.3989639, -196.8578796, 196.8459167
30: -102.9253006, 80.1761322, -102.9363327, 80.1991196, -183.1244202, 183.1124573
31: -106.6536560, 67.5429840, -106.6663971, 67.5653992, -174.2190552, 174.2093811
32: -100.1512756, 73.7404404, -100.1671677, 73.7537308, -173.9049988, 173.9076080
33: -141.1815643, 80.9286804, -141.2058563, 80.9379120, -222.1194763, 222.1345215
34: -120.1827469, 73.0269165, -120.2015152, 73.0394821, -193.2222290, 193.2284241
35: -120.7642517, 70.4415741, -120.7873688, 70.4483871, -191.2126312, 191.2289276
36: -117.9226379, 69.8307648, -117.9452972, 69.8383789, -187.7610168, 187.7760620
37: -164.8338623, 74.2456055, -164.8552551, 74.2686615, -239.1025085, 239.1008606
38: -145.9229126, 86.4525146, -145.9501190, 86.4607544, -232.3836670, 232.4026184
39: -168.5646667, 78.1052399, -168.5904541, 78.1125870, -246.6772461, 246.6956940
40: -135.5876770, 73.8635330, -135.6093750, 73.8960037, -209.4836731, 209.4729004
41: -100.8066711, 67.4122086, -100.8195724, 67.4481964, -168.2548676, 168.2317810
42: -75.8448944, 65.9455109, -75.8553162, 65.9716949, -141.8165894, 141.8008270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0705068, upper bound: 99.0718345
time: 107.13 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0705068, upper bound: 99.1343693
time: 94.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 203.92 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 203.92
Output dim: 5, lower bound: -99.0705068, upper bound: 99.0717147
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 203.92
Output dim: 5, lower bound: -99.0705068, upper bound: 99.1340867
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 203.92
Output dim: 5, lower bound: -99.0705068, upper bound: 99.0718345
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 203.92
Output dim: 5, lower bound: -99.0705068, upper bound: 99.1343693

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -124.4803238, 84.2534866, -124.6823502, 84.3527145, -208.8330078, 208.9358368
1: -69.8480682, 74.2014694, -69.9894791, 74.2994003, -144.1474609, 144.1909485
2: -62.5971527, 71.1310425, -62.7464218, 71.3044891, -133.9016266, 133.8774567
3: -72.0137863, 86.0586700, -72.2134552, 86.3072968, -158.3210754, 158.2721252
4: -75.1112518, 84.4221191, -75.3119659, 84.5865707, -159.6978149, 159.7340698
5: -67.3672714, 90.4528046, -67.5240479, 90.6890259, -158.0563049, 157.9768524
6: -102.5043106, 75.6259689, -102.5931244, 75.7536774, -178.2579956, 178.2190857
7: -83.3625488, 91.1365051, -83.5107117, 91.2378998, -174.6004486, 174.6472168
8: -88.3760376, 101.4760971, -88.5499496, 101.6756821, -190.0517273, 190.0260468
9: -78.0897064, 81.5780869, -78.1912384, 81.6980438, -159.7877502, 159.7693176
10: -110.6778336, 117.0800476, -111.0559387, 117.5311356, -228.2089233, 228.1359863
11: -110.4912186, 82.9450150, -110.8440018, 83.2541962, -193.7453918, 193.7890167
12: -110.8876572, 88.4883575, -111.2095032, 88.7914429, -199.6791077, 199.6978607
13: -109.7397003, 100.0608521, -109.9903793, 100.3388214, -210.0785217, 210.0512390
14: -162.5248871, 83.3049393, -162.8523865, 83.5741272, -246.0990143, 246.1573181
15: -91.3264313, 81.4982452, -91.5540695, 81.5548553, -172.8812866, 173.0523071
16: -117.9562759, 97.0711746, -118.1015778, 97.3146057, -215.2708740, 215.1727600
17: -164.0531464, 118.8015594, -164.3867798, 119.1587906, -283.2119141, 283.1883545
18: -101.4203796, 84.1952820, -101.6787796, 84.4465637, -185.8669434, 185.8740540
19: -84.8996735, 47.3387299, -85.1507721, 47.4657974, -132.3654480, 132.4895020
20: -74.5191193, 57.2931862, -74.7157593, 57.4036560, -131.9227753, 132.0089417
21: -104.2543488, 62.8249168, -104.5621948, 63.0083809, -167.2627258, 167.3871002
22: -113.0127869, 72.7172165, -113.1638336, 72.8434677, -185.8562622, 185.8810425
23: -86.2126999, 58.1361961, -86.4110641, 58.3065071, -144.5192108, 144.5472565
24: -103.3108978, 69.0345001, -103.5013123, 69.1491089, -172.4600067, 172.5358124
25: -90.7659836, 67.8478012, -90.9050140, 67.9662247, -158.7322083, 158.7527924
26: -121.8611450, 89.1829224, -122.1831665, 89.4435883, -211.3047333, 211.3660736
27: -104.1718216, 73.8449860, -104.3067627, 73.9193497, -178.0911713, 178.1517334
28: -85.4558487, 62.9415932, -85.5897293, 63.0129013, -148.4687500, 148.5313263
29: -119.0960846, 76.3426590, -119.2533569, 76.4911194, -195.5872040, 195.5960083
30: -102.4977875, 79.2034225, -102.7089081, 79.4036179, -181.9013977, 181.9123230
31: -106.0576706, 66.7034607, -106.3623199, 66.9050598, -172.9627380, 173.0657654
32: -99.7668228, 73.2641754, -99.8916931, 73.3443832, -173.1112061, 173.1558533
33: -140.2362518, 80.5104218, -140.4651794, 80.7139435, -220.9501953, 220.9755859
34: -119.4997177, 72.6524353, -119.6976929, 72.7652130, -192.2649078, 192.3501282
35: -119.9030838, 70.1260300, -120.1602097, 70.2753220, -190.1783905, 190.2862396
36: -117.2435989, 69.5691071, -117.4756165, 69.6535492, -186.8971558, 187.0447235
37: -164.2988281, 73.8151093, -164.4754028, 73.9109192, -238.2097321, 238.2905121
38: -145.0449677, 86.0569305, -145.2983398, 86.2279434, -231.2729034, 231.3552704
39: -167.6472778, 77.8024139, -167.8927917, 77.9447861, -245.5920563, 245.6952057
40: -134.9252014, 73.6530304, -135.1310120, 73.7377548, -208.6629639, 208.7840424
41: -100.4144058, 67.0068283, -100.5310135, 67.1160049, -167.5304108, 167.5378418
42: -75.5295410, 65.0918427, -75.6358566, 65.3083344, -140.8378601, 140.7276917

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=678, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 647

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0597055, upper bound: 99.0049765
time: 92.22 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0597055, upper bound: 99.0626881
time: 105.12 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -124.5763321, 84.2763290, -125.0160217, 84.5377808, -209.1141052, 209.2923584
1: -69.9109879, 74.2179108, -70.2042084, 74.4446564, -144.3556366, 144.4221039
2: -62.6994667, 71.1471558, -63.0478134, 71.6139450, -134.3134155, 134.1949768
3: -72.1337738, 86.0799637, -72.5673676, 86.6881714, -158.8219452, 158.6473389
4: -75.2268524, 84.4402161, -75.6536865, 84.8678436, -160.0946960, 160.0939026
5: -67.4739914, 90.4721680, -67.8394470, 91.0815277, -158.5555115, 158.3116150
6: -102.5356369, 75.6328125, -102.7876129, 75.8948364, -178.4304810, 178.4204254
7: -83.4446182, 91.1526947, -83.7945557, 91.4067001, -174.8513184, 174.9472351
8: -88.4797516, 101.4967270, -88.8665771, 101.9776077, -190.4573669, 190.3633118
9: -78.1461334, 81.6076584, -78.4140472, 81.8724213, -160.0185547, 160.0216980
10: -110.7222900, 117.2202606, -111.3741684, 118.0032730, -228.7255554, 228.5944214
11: -110.5202026, 83.1386185, -111.3491592, 83.7832108, -194.3034058, 194.4877777
12: -110.9133301, 88.6665726, -111.7324982, 89.3087769, -200.2221069, 200.3990784
13: -109.8019104, 100.1082153, -110.2642365, 100.7219696, -210.5238800, 210.3724518
14: -162.5828247, 83.4514999, -163.2879181, 83.9889984, -246.5718231, 246.7394104
15: -91.3374023, 81.5268402, -91.7696152, 81.7516098, -173.0890198, 173.2964478
16: -118.0101089, 97.0753098, -118.4230270, 97.4833832, -215.4934998, 215.4983368
17: -164.0990753, 119.0225372, -164.9152222, 119.7872543, -283.8863220, 283.9377441
18: -101.4657288, 84.3403931, -102.1226501, 84.8835678, -186.3493042, 186.4630432
19: -84.9279099, 47.4279251, -85.5156555, 47.7231064, -132.6510162, 132.9435730
20: -74.5483093, 57.3674965, -74.9944458, 57.6263466, -132.1746521, 132.3619385
21: -104.2835541, 62.9646568, -105.0183258, 63.3921738, -167.6757202, 167.9829865
22: -113.0421448, 72.8288116, -113.3931274, 73.1884460, -186.2305908, 186.2219391
23: -86.2403183, 58.2234344, -86.7123032, 58.5660210, -144.8063354, 144.9357300
24: -103.3402176, 69.1095276, -103.7578812, 69.3739777, -172.7141876, 172.8674011
25: -90.7889328, 67.9150543, -91.0707703, 68.1822357, -158.9711609, 158.9858246
26: -121.8931427, 89.3633423, -122.7182999, 89.9642258, -211.8573608, 212.0816193
27: -104.2151031, 73.9424133, -104.6359253, 74.2034302, -178.4185333, 178.5783386
28: -85.4858932, 63.0109711, -85.8856964, 63.2347755, -148.7206573, 148.8966675
29: -119.1221695, 76.5025635, -119.5316315, 76.9326019, -196.0547638, 196.0341949
30: -102.5225754, 79.3322830, -103.0290146, 79.7944870, -182.3170471, 182.3612976
31: -106.0997009, 66.7928238, -106.7778320, 67.1698074, -173.2695007, 173.5706482
32: -99.7988129, 73.3376617, -100.0987625, 73.5842896, -173.3830719, 173.4364319
33: -140.3496704, 80.5403366, -140.8156128, 80.9901428, -221.3398132, 221.3559265
34: -119.5696106, 72.6881790, -119.9471970, 72.9615173, -192.5311279, 192.6353760
35: -119.9797668, 70.1465683, -120.4201813, 70.4425507, -190.4223175, 190.5667419
36: -117.2877350, 69.5939560, -117.6718369, 69.7828827, -187.0706177, 187.2657928
37: -164.3423157, 73.8599930, -164.7364960, 74.1079559, -238.4502411, 238.5964661
38: -145.1294708, 86.0782166, -145.6361237, 86.3895493, -231.5190125, 231.7143402
39: -167.7425232, 77.8250732, -168.2236176, 78.1707153, -245.9132385, 246.0486755
40: -134.9781494, 73.6399536, -135.3595886, 73.8256989, -208.8038483, 208.9995422
41: -100.4492874, 67.0261993, -100.7277298, 67.2748566, -167.7241516, 167.7539368
42: -75.5529633, 65.1940231, -75.8383179, 65.6439362, -141.1968689, 141.0323486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=679, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 647

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0607059, upper bound: 99.0693306
time: 109.62 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0607059, upper bound: 99.1250549
time: 132.51 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -125.1508789, 84.5006104, -125.0003204, 84.4590607, -209.6099396, 209.5009155
1: -70.3195038, 74.3990021, -70.2172470, 74.3733444, -144.6928406, 144.6162415
2: -63.2201385, 71.4035492, -63.0575790, 71.3766479, -134.5967865, 134.4611206
3: -72.7913055, 86.4484863, -72.6041260, 86.4120102, -159.2033081, 159.0526123
4: -75.8624878, 84.7107468, -75.6815338, 84.6781769, -160.5406494, 160.3922729
5: -67.9955444, 90.8167191, -67.8318939, 90.7837143, -158.7792664, 158.6486206
6: -102.7970657, 75.9944000, -102.7415085, 75.9333801, -178.7304230, 178.7359009
7: -83.9145508, 91.3426514, -83.7747345, 91.3162537, -175.2308044, 175.1173859
8: -89.0561066, 101.8111801, -88.8903503, 101.7762756, -190.8323669, 190.7015381
9: -78.4593964, 81.9118042, -78.3643341, 81.8564606, -160.3158569, 160.2761383
10: -111.3184738, 118.3929901, -111.2387238, 118.1842194, -229.5026855, 229.6317139
11: -111.0464859, 84.1822891, -110.9870148, 83.8877335, -194.9342194, 195.1693115
12: -111.3710175, 89.6318970, -111.3266220, 89.3585205, -200.7295380, 200.9585114
13: -110.5881195, 100.6153336, -110.4127579, 100.5335159, -211.1216125, 211.0280914
14: -163.1664124, 84.3099060, -163.0720825, 84.0834808, -247.2498932, 247.3819885
15: -91.9551239, 81.7406464, -91.8470917, 81.6844482, -173.6395721, 173.5877380
16: -118.4279633, 97.7689819, -118.3299561, 97.6613846, -216.0893250, 216.0989380
17: -164.6242828, 120.2527390, -164.5474548, 119.8996201, -284.5238647, 284.8001404
18: -101.9475021, 85.1901245, -101.8707047, 84.9526367, -186.9001160, 187.0608215
19: -85.3136978, 47.8880959, -85.2652817, 47.7478790, -133.0615540, 133.1533813
20: -74.8940277, 57.7572021, -74.8433533, 57.6388054, -132.5328369, 132.6005554
21: -104.7394485, 63.6287918, -104.6876526, 63.4162369, -168.1556854, 168.3164368
22: -113.3348999, 73.3601837, -113.2981796, 73.1685257, -186.5034180, 186.6583557
23: -86.5607529, 58.7462158, -86.5171051, 58.6119347, -145.1726837, 145.2633057
24: -103.6831207, 69.5148621, -103.6355286, 69.3951950, -173.0783081, 173.1503906
25: -91.0549164, 68.3330460, -91.0130539, 68.2140656, -159.2689819, 159.3460999
26: -122.3912811, 90.2540207, -122.3352814, 89.9732437, -212.3645172, 212.5892944
27: -104.5650482, 74.3184052, -104.4926376, 74.1600418, -178.7250671, 178.8110352
28: -85.7513733, 63.3147469, -85.7037964, 63.1987228, -148.9500885, 149.0185242
29: -119.4151077, 77.1597443, -119.3715973, 76.9048386, -196.3199158, 196.5313263
30: -102.8802109, 80.0057449, -102.8322067, 79.8036346, -182.6838379, 182.8379364
31: -106.5937500, 67.4173279, -106.5276871, 67.2743530, -173.8681030, 173.9450073
32: -100.0975189, 73.6432800, -100.0424500, 73.5296860, -173.6271973, 173.6857300
33: -141.0316162, 80.8801117, -140.8574829, 80.8255463, -221.8571625, 221.7375946
34: -120.0884628, 72.9647827, -119.9828415, 72.8963699, -192.9848328, 192.9476318
35: -120.6524124, 70.4063492, -120.5322037, 70.3669739, -191.0193787, 190.9385376
36: -117.8504257, 69.7884750, -117.7786789, 69.7410660, -187.5914917, 187.5671539
37: -164.7636108, 74.1586609, -164.6923828, 74.0679092, -238.8314972, 238.8510437
38: -145.8014832, 86.4023666, -145.6692963, 86.3457336, -232.1472168, 232.0716553
39: -168.4304504, 78.0706024, -168.2792358, 78.0326538, -246.4630737, 246.3498383
40: -135.5019531, 73.8188782, -135.4100952, 73.7939148, -209.2958374, 209.2289734
41: -100.7499847, 67.3316803, -100.6880722, 67.2623291, -168.0122986, 168.0197449
42: -75.8011093, 65.8101349, -75.7541351, 65.6582642, -141.4593658, 141.5642700

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=679, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0170438, upper bound: 99.0569100
time: 93.11 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0170438, upper bound: 99.0665439
time: 101.15 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -125.2515335, 84.5266266, -125.3381348, 84.6429138, -209.8944397, 209.8647614
1: -70.3876953, 74.4165115, -70.4359207, 74.5189209, -144.9066162, 144.8524170
2: -63.3283386, 71.4208069, -63.3625717, 71.6865311, -135.0148621, 134.7833557
3: -72.9178543, 86.4710236, -72.9612503, 86.7933044, -159.7111511, 159.4322662
4: -75.9822845, 84.7321472, -76.0272522, 84.9598923, -160.9421692, 160.7593994
5: -68.1063232, 90.8375397, -68.1509247, 91.1765671, -159.2828827, 158.9884644
6: -102.8325195, 76.0076141, -102.9393768, 76.0769806, -178.9094696, 178.9469910
7: -84.0036926, 91.3598709, -84.0656433, 91.4852600, -175.4889221, 175.4255066
8: -89.1665421, 101.8331757, -89.2108231, 102.0790710, -191.2456055, 191.0440063
9: -78.5294571, 81.9441376, -78.5945282, 82.0330582, -160.5625153, 160.5386658
10: -111.3703308, 118.5399017, -111.5572891, 118.6625824, -230.0329132, 230.0971832
11: -111.0787430, 84.3808289, -111.4916077, 84.4210358, -195.4997406, 195.8724365
12: -111.3997498, 89.8141174, -111.8496552, 89.8796692, -201.2794037, 201.6637726
13: -110.6765213, 100.6642838, -110.7070236, 100.9198227, -211.5963135, 211.3713074
14: -163.2274780, 84.4636765, -163.5084381, 84.5004730, -247.7279510, 247.9721069
15: -91.9850235, 81.7730026, -92.0739670, 81.8845291, -173.8695526, 173.8469696
16: -118.4885406, 97.7786560, -118.6560364, 97.8354492, -216.3239899, 216.4346924
17: -164.6729736, 120.4863129, -165.0765076, 120.5318985, -285.2048645, 285.5628052
18: -101.9954453, 85.3488159, -102.3122787, 85.3930664, -187.3884888, 187.6611023
19: -85.3436737, 47.9839706, -85.6314850, 48.0084610, -133.3521423, 133.6154480
20: -74.9251251, 57.8368912, -75.1227875, 57.8634605, -132.7885895, 132.9596558
21: -104.7708664, 63.7740784, -105.1453171, 63.8031273, -168.5739899, 168.9193726
22: -113.3659515, 73.4844818, -113.5279083, 73.5218353, -186.8877869, 187.0123901
23: -86.5895157, 58.8381004, -86.8189240, 58.8741150, -145.4636230, 145.6570129
24: -103.7132492, 69.5974503, -103.8925781, 69.6212006, -173.3344421, 173.4900208
25: -91.0787354, 68.4095383, -91.1799240, 68.4341736, -159.5129089, 159.5894470
26: -122.4254074, 90.4428482, -122.8709717, 90.4987793, -212.9241943, 213.3138123
27: -104.6100616, 74.4240723, -104.8228226, 74.4459152, -179.0559692, 179.2468872
28: -85.7827606, 63.3915787, -86.0001221, 63.4235573, -149.2063141, 149.3916931
29: -119.4427490, 77.3282318, -119.6504822, 77.3524780, -196.7952271, 196.9787140
30: -102.9070816, 80.1417465, -103.1536255, 80.1982574, -183.1053467, 183.2953491
31: -106.6362610, 67.5165253, -106.9423981, 67.5423508, -174.1786041, 174.4589233
32: -100.1337128, 73.7191544, -100.2496796, 73.7715607, -173.9052734, 173.9688416
33: -141.1484985, 80.9132233, -141.2100677, 81.1030121, -222.2514954, 222.1232910
34: -120.1609726, 73.0076904, -120.2341690, 73.0964050, -193.2573853, 193.2418518
35: -120.7340240, 70.4306107, -120.7966080, 70.5365143, -191.2705383, 191.2272186
36: -117.8976440, 69.8172531, -117.9776382, 69.8743668, -187.7720032, 187.7948914
37: -164.8109283, 74.2069016, -164.9551697, 74.2654648, -239.0763855, 239.1620483
38: -145.8918762, 86.4283600, -146.0120087, 86.5095215, -232.4013672, 232.4403687
39: -168.5305481, 78.0943909, -168.6140747, 78.2594070, -246.7899475, 246.7084656
40: -135.5599976, 73.8114929, -135.6416626, 73.8887405, -209.4487305, 209.4531555
41: -100.7903976, 67.3487396, -100.8860092, 67.4207001, -168.2110901, 168.2347412
42: -75.8314056, 65.9120483, -75.9570007, 65.9955139, -141.8269196, 141.8690338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0170438, upper bound: 99.0569100
time: 112.87 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0170438, upper bound: 99.0665439
time: 144.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 259.79 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 259.79
Output dim: 5, lower bound: -99.0597055, upper bound: 99.0049765
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 259.79
Output dim: 5, lower bound: -99.0597055, upper bound: 99.0626881
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 259.79
Output dim: 5, lower bound: -99.0607059, upper bound: 99.0693306
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 259.79
Output dim: 5, lower bound: -99.0607059, upper bound: 99.1250549
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 259.79
Output dim: 5, lower bound: -99.0170438, upper bound: 99.0569100
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 259.79
Output dim: 5, lower bound: -99.0170438, upper bound: 99.0665439
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 259.79
Output dim: 5, lower bound: -99.0170438, upper bound: 99.0569100
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 259.79
Output dim: 5, lower bound: -99.0170438, upper bound: 99.0665439

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -124.2242432, 83.9199524, -124.6312408, 84.2062988, -208.4305420, 208.5511932
1: -69.6643143, 73.9745941, -69.9586792, 74.2012939, -143.8656006, 143.9332733
2: -62.3878860, 70.8746185, -62.7163887, 71.1874771, -133.5753632, 133.5910034
3: -71.7615967, 85.6980743, -72.1848602, 86.1452484, -157.9068298, 157.8829193
4: -74.9338684, 84.3061676, -75.2762451, 84.5427704, -159.4766388, 159.5824127
5: -67.0876312, 90.0514297, -67.4886475, 90.5083771, -157.5959930, 157.5400696
6: -102.3494492, 75.2626801, -102.5532532, 75.6029053, -177.9523621, 177.8159180
7: -82.9975891, 90.6522064, -83.4568100, 91.0134964, -174.0110779, 174.1090088
8: -88.1608734, 101.2006607, -88.5230255, 101.5540085, -189.7148743, 189.7236786
9: -77.8523102, 81.3729706, -78.0910187, 81.6666107, -159.5189056, 159.4639893
10: -110.2976456, 116.7305374, -110.8939285, 117.4877167, -227.7853546, 227.6244659
11: -110.3357620, 82.7247009, -110.7719650, 83.2168732, -193.5525970, 193.4966736
12: -110.2217026, 88.0217133, -110.8978119, 88.7507324, -198.9724426, 198.9195099
13: -109.3654633, 99.7822189, -109.8403015, 100.2687912, -209.6342468, 209.6225281
14: -161.9072571, 82.9740524, -162.5908051, 83.5502853, -245.4575195, 245.5648499
15: -90.8345871, 81.2492371, -91.3583069, 81.5084686, -172.3430481, 172.6075439
16: -117.6872406, 96.7566986, -118.0279007, 97.2132263, -214.9004669, 214.7845917
17: -163.5983429, 118.4580917, -164.1895752, 119.1208420, -282.7191772, 282.6476746
18: -101.1804810, 84.0691147, -101.5885162, 84.4173889, -185.5978699, 185.6576080
19: -84.7799683, 47.2771873, -85.1017609, 47.4492683, -132.2292328, 132.3789520
20: -74.3376770, 57.2047310, -74.6513367, 57.3824234, -131.7200928, 131.8560638
21: -104.1098785, 62.7059860, -104.4947052, 62.9826698, -167.0925293, 167.2006836
22: -112.4691772, 72.3955612, -112.9260330, 72.7992477, -185.2684326, 185.3215942
23: -86.0786514, 58.0289116, -86.3623886, 58.2723656, -144.3510132, 144.3912964
24: -103.1477509, 68.9539337, -103.4516907, 69.1227188, -172.2704773, 172.4056244
25: -90.5341949, 67.6669464, -90.8086243, 67.9252243, -158.4594116, 158.4755707
26: -121.1361618, 88.7768784, -121.8618164, 89.4078827, -210.5440369, 210.6386719
27: -103.9312897, 73.6904221, -104.2548523, 73.8575592, -177.7888489, 177.9452820
28: -85.3185730, 62.8377075, -85.5493164, 62.9763260, -148.2948914, 148.3870239
29: -118.7147446, 76.0231323, -119.0881119, 76.4517593, -195.1665039, 195.1112366
30: -102.3405304, 78.9896088, -102.6586075, 79.3289032, -181.6694336, 181.6482239
31: -105.8919373, 66.5882950, -106.3042450, 66.8713760, -172.7633057, 172.8925171
32: -99.5707245, 73.1004333, -99.8159790, 73.3065033, -172.8772278, 172.9164124
33: -140.0121460, 80.3753815, -140.4197388, 80.6594696, -220.6716003, 220.7951202
34: -119.2593536, 72.4690933, -119.6500168, 72.6903534, -191.9497070, 192.1190948
35: -119.7025375, 69.9832840, -120.1192398, 70.2168045, -189.9193115, 190.1025238
36: -117.0603027, 69.4656830, -117.4176483, 69.6100769, -186.6703644, 186.8833313
37: -164.0573730, 73.7106018, -164.3961487, 73.8785782, -237.9359131, 238.1067505
38: -144.8230896, 85.9253998, -145.2516174, 86.1769257, -231.0000153, 231.1770172
39: -167.4098969, 77.6768036, -167.8187103, 77.8970337, -245.3069153, 245.4955139
40: -134.7083893, 73.4085770, -135.0837402, 73.6259842, -208.3343811, 208.4923096
41: -100.2656403, 66.7497101, -100.4933624, 67.0104446, -167.2760925, 167.2430725
42: -75.4487610, 64.8803940, -75.5986938, 65.2566299, -140.7053833, 140.4790649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=678, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0411774, upper bound: 98.9449994
time: 860.39 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0411774, upper bound: 98.9449994
time: 304.71 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -124.4563446, 84.2359314, -124.6723175, 84.3451691, -208.8015137, 208.9082336
1: -69.8339233, 74.1897049, -69.9835510, 74.2945404, -144.1284637, 144.1732330
2: -62.5828018, 71.1209641, -62.7404022, 71.3003159, -133.8831177, 133.8613586
3: -72.0004120, 86.0428162, -72.2078629, 86.3007202, -158.3011322, 158.2506714
4: -75.0962448, 84.4118423, -75.3056946, 84.5823212, -159.6785431, 159.7175293
5: -67.3551483, 90.4370270, -67.5189209, 90.6825256, -158.0376740, 157.9559479
6: -102.4893036, 75.5722961, -102.5868073, 75.7316895, -178.2209778, 178.1590881
7: -83.3453140, 91.1206207, -83.5034332, 91.2314148, -174.5767212, 174.6240540
8: -88.3611298, 101.4642410, -88.5437164, 101.6707230, -190.0318604, 190.0079498
9: -78.0752411, 81.5677338, -78.1851578, 81.6935883, -159.7688293, 159.7528839
10: -110.6578140, 117.0599823, -111.0474777, 117.5227051, -228.1805115, 228.1074524
11: -110.4688950, 82.8494110, -110.8345947, 83.2143250, -193.6832275, 193.6839905
12: -110.8652802, 88.4729004, -111.2003098, 88.7850037, -199.6502533, 199.6732178
13: -109.6982574, 100.0389481, -109.9723282, 100.3295135, -210.0277710, 210.0112762
14: -162.4995728, 83.2986603, -162.8419647, 83.5714722, -246.0710449, 246.1406250
15: -91.2634811, 81.4829712, -91.5283432, 81.5483551, -172.8118286, 173.0112915
16: -117.9313889, 96.9800873, -118.0909958, 97.2790070, -215.2103882, 215.0710754
17: -164.0363922, 118.7779770, -164.3798828, 119.1490631, -283.1854248, 283.1578369
18: -101.4051666, 84.1820221, -101.6720963, 84.4409943, -185.8461609, 185.8541260
19: -84.8897400, 47.3284378, -85.1465302, 47.4613838, -132.3511200, 132.4749756
20: -74.5073242, 57.2835350, -74.7108765, 57.3995895, -131.9069214, 131.9944153
21: -104.2395401, 62.8066177, -104.5560150, 63.0007095, -167.2402496, 167.3626404
22: -112.9661865, 72.6976624, -113.1445618, 72.8351593, -185.8013458, 185.8422241
23: -86.2036133, 58.1030807, -86.4072266, 58.2926636, -144.4962769, 144.5102997
24: -103.2938232, 69.0258789, -103.4940720, 69.1454773, -172.4393005, 172.5199585
25: -90.7478485, 67.8351517, -90.8966827, 67.9609070, -158.7087555, 158.7318115
26: -121.8366241, 89.1691208, -122.1731186, 89.4377289, -211.2743225, 211.3422394
27: -104.1529694, 73.8276215, -104.2988892, 73.9105225, -178.0634766, 178.1265106
28: -85.4485931, 62.9282112, -85.5866928, 63.0071716, -148.4557648, 148.5148926
29: -119.0685120, 76.3248901, -119.2418442, 76.4835815, -195.5520630, 195.5667267
30: -102.4806366, 79.1593399, -102.7017441, 79.3854828, -181.8661194, 181.8610840
31: -106.0441971, 66.6596909, -106.3563995, 66.8870773, -172.9312744, 173.0160828
32: -99.7500153, 73.2503204, -99.8847504, 73.3385315, -173.0885468, 173.1350708
33: -140.2209930, 80.4965668, -140.4587402, 80.7081909, -220.9291840, 220.9553070
34: -119.4888992, 72.6338959, -119.6930771, 72.7573929, -192.2462921, 192.3269653
35: -119.8879852, 70.1138763, -120.1538544, 70.2702026, -190.1581879, 190.2677307
36: -117.2140274, 69.5589600, -117.4634399, 69.6492081, -186.8632355, 187.0223999
37: -164.2693634, 73.8050079, -164.4630127, 73.9066772, -238.1760254, 238.2680206
38: -145.0312195, 86.0454712, -145.2925415, 86.2231369, -231.2543640, 231.3379974
39: -167.5949249, 77.7930908, -167.8712463, 77.9409561, -245.5358887, 245.6643372
40: -134.9040527, 73.6372986, -135.1221313, 73.7312927, -208.6353455, 208.7593994
41: -100.4013977, 66.9740906, -100.5255585, 67.1025467, -167.5039368, 167.4996490
42: -75.5189438, 65.0424728, -75.6313705, 65.2879257, -140.8068542, 140.6738434

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=678, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0411774, upper bound: 98.9939112
time: 177.83 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0411774, upper bound: 99.0573428
time: 125.70 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -124.3200302, 83.9427795, -124.9644928, 84.3911209, -208.7111359, 208.9072723
1: -69.7270737, 73.9911499, -70.1731033, 74.3464432, -144.0735168, 144.1642456
2: -62.4900475, 70.8908157, -63.0177612, 71.4968643, -133.9869080, 133.9085693
3: -71.8814545, 85.7195435, -72.5387421, 86.5260773, -158.4075317, 158.2582703
4: -75.0492859, 84.3242645, -75.6178436, 84.8236771, -159.8729553, 159.9420929
5: -67.1942978, 90.0708847, -67.8039551, 90.9008636, -158.0951538, 157.8748474
6: -102.3805695, 75.2679443, -102.7468567, 75.7433548, -178.1239166, 178.0148010
7: -83.0792999, 90.6687698, -83.7397766, 91.1822510, -174.2615509, 174.4085236
8: -88.2644653, 101.2213821, -88.8395996, 101.8556824, -190.1201477, 190.0609741
9: -77.9123001, 81.4022751, -78.3192902, 81.8406982, -159.7529907, 159.7215576
10: -110.3423996, 116.8705521, -111.2115631, 117.9598236, -228.3022156, 228.0820923
11: -110.3632584, 82.9183350, -111.2756042, 83.7456665, -194.1089172, 194.1939392
12: -110.2475510, 88.1997299, -111.4206924, 89.2680664, -199.5156250, 199.6204224
13: -109.4399796, 99.8292160, -110.1249084, 100.6508408, -210.0908203, 209.9541016
14: -161.9652710, 83.1204987, -163.0259399, 83.9648895, -245.9301605, 246.1464386
15: -90.8455353, 81.2776031, -91.5735168, 81.7041168, -172.5496521, 172.8511200
16: -117.7406464, 96.7606125, -118.3483582, 97.3822479, -215.1228943, 215.1089783
17: -163.6443481, 118.6787872, -164.7178650, 119.7491379, -283.3934937, 283.3966064
18: -101.2249374, 84.2142410, -102.0294495, 84.8544312, -186.0793457, 186.2436829
19: -84.8083725, 47.3663559, -85.4663315, 47.7064934, -132.5148621, 132.8326874
20: -74.3665924, 57.2789497, -74.9296036, 57.6049919, -131.9715881, 132.2085419
21: -104.1391449, 62.8458328, -104.9503250, 63.3663750, -167.5055237, 167.7961578
22: -112.4987411, 72.5066071, -113.1550674, 73.1430817, -185.6418152, 185.6616821
23: -86.1062164, 58.1161842, -86.6632690, 58.5317764, -144.6379852, 144.7794495
24: -103.1767426, 69.0289612, -103.7076569, 69.3475037, -172.5242462, 172.7366028
25: -90.5571213, 67.7339478, -90.9742508, 68.1405716, -158.6976776, 158.7081604
26: -121.1686935, 88.9573593, -122.3968658, 89.9284058, -211.0970917, 211.3542175
27: -103.9739990, 73.7880707, -104.5831451, 74.1415253, -178.1155243, 178.3712158
28: -85.3484802, 62.9071846, -85.8449707, 63.1980133, -148.5464935, 148.7521515
29: -118.7409744, 76.1826477, -119.3661194, 76.8924255, -195.6333923, 195.5487671
30: -102.3649521, 79.1188583, -102.9782181, 79.7198257, -182.0847778, 182.0970764
31: -105.9332733, 66.6775742, -106.7184601, 67.1359253, -173.0691986, 173.3960266
32: -99.6027145, 73.1737366, -100.0225372, 73.5462799, -173.1489716, 173.1962738
33: -140.1253357, 80.4053650, -140.7699432, 80.9348450, -221.0601807, 221.1753082
34: -119.3290024, 72.5049210, -119.8993301, 72.8865814, -192.2155762, 192.4042358
35: -119.7792358, 70.0039597, -120.3795319, 70.3837738, -190.1629944, 190.3834839
36: -117.1044769, 69.4904175, -117.6138611, 69.7389221, -186.8433990, 187.1042633
37: -164.1008453, 73.7561417, -164.6567841, 74.0757523, -238.1765747, 238.4129333
38: -144.9074097, 85.9466629, -145.5892944, 86.3383865, -231.2457886, 231.5359497
39: -167.5051880, 77.6989288, -168.1494751, 78.1224747, -245.6276398, 245.8483887
40: -134.7610931, 73.3954010, -135.3114166, 73.7142563, -208.4753418, 208.7067871
41: -100.3003464, 66.7687683, -100.6894836, 67.1682663, -167.4686127, 167.4582520
42: -75.4719543, 64.9828491, -75.8001785, 65.5921631, -141.0641174, 140.7830200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=678, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0411774, upper bound: 99.0163707
time: 102.59 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0411774, upper bound: 99.0596552
time: 108.00 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -124.5522842, 84.2587585, -125.0054169, 84.5301285, -209.0823975, 209.2641754
1: -69.8967819, 74.2061462, -70.1978836, 74.4396820, -144.3364563, 144.4040222
2: -62.6850815, 71.1370697, -63.0415993, 71.6096802, -134.2947540, 134.1786499
3: -72.1203842, 86.0641022, -72.5616074, 86.6814117, -158.8017883, 158.6257019
4: -75.2117615, 84.4299774, -75.6471558, 84.8632965, -160.0750427, 160.0771332
5: -67.4618530, 90.4563599, -67.8340149, 91.0749435, -158.5368042, 158.2903748
6: -102.5206375, 75.5799255, -102.7805328, 75.8729858, -178.3935852, 178.3604584
7: -83.4273376, 91.1368103, -83.7863083, 91.4000702, -174.8274078, 174.9230957
8: -88.4648438, 101.4849091, -88.8601151, 101.9724579, -190.4372864, 190.3450317
9: -78.1315918, 81.5973129, -78.4062500, 81.8678055, -159.9993896, 160.0035553
10: -110.7022858, 117.2001266, -111.3651886, 117.9945374, -228.6968231, 228.5653076
11: -110.4979706, 83.0427399, -111.3392792, 83.7433014, -194.2412567, 194.3820190
12: -110.8909531, 88.6510773, -111.7231979, 89.3021088, -200.1930542, 200.3742371
13: -109.7604904, 100.0862503, -110.2418518, 100.7118607, -210.4723358, 210.3280945
14: -162.5574646, 83.4452515, -163.2773438, 83.9860458, -246.5435028, 246.7225800
15: -91.2756195, 81.5115662, -91.7439423, 81.7442474, -173.0198364, 173.2554932
16: -117.9852524, 96.9844208, -118.4115601, 97.4479523, -215.4331970, 215.3959808
17: -164.0823517, 118.9988251, -164.9081726, 119.7771378, -283.8594971, 283.9069824
18: -101.4504547, 84.3271332, -102.1151505, 84.8777390, -186.3281860, 186.4422607
19: -84.9179382, 47.4176636, -85.5111771, 47.7185211, -132.6364594, 132.9288330
20: -74.5365601, 57.3578072, -74.9893570, 57.6221466, -132.1586914, 132.3471680
21: -104.2687607, 62.9464378, -105.0118179, 63.3843040, -167.6530609, 167.9582520
22: -112.9953690, 72.8092041, -113.3737030, 73.1788940, -186.1742554, 186.1829071
23: -86.2312469, 58.1902695, -86.7082901, 58.5520630, -144.7832947, 144.8985596
24: -103.3230896, 69.1009216, -103.7504272, 69.3702316, -172.6933136, 172.8513489
25: -90.7707443, 67.9023895, -91.0623016, 68.1763611, -158.9471130, 158.9646912
26: -121.8685837, 89.3495255, -122.7081299, 89.9578705, -211.8264465, 212.0576324
27: -104.1961365, 73.9249954, -104.6276779, 74.1944199, -178.3905487, 178.5526733
28: -85.4786301, 62.9975548, -85.8825684, 63.2289505, -148.7075806, 148.8801270
29: -119.0945435, 76.4846954, -119.5199203, 76.9242706, -196.0187988, 196.0046082
30: -102.5054932, 79.2882156, -103.0215683, 79.7761154, -182.2816010, 182.3097839
31: -106.0862045, 66.7490387, -106.7715225, 67.1515808, -173.2377930, 173.5205383
32: -99.7819366, 73.3237839, -100.0915070, 73.5782776, -173.3602142, 173.4152832
33: -140.3344116, 80.5264893, -140.8090210, 80.9839478, -221.3183441, 221.3355103
34: -119.5588150, 72.6695862, -119.9424667, 72.9534760, -192.5122986, 192.6120605
35: -119.9646378, 70.1344147, -120.4136581, 70.4369965, -190.4016266, 190.5480652
36: -117.2582550, 69.5837708, -117.6594772, 69.7778549, -187.0361023, 187.2432556
37: -164.3128815, 73.8499146, -164.7237549, 74.1035767, -238.4164581, 238.5736694
38: -145.1157074, 86.0668182, -145.6300049, 86.3845367, -231.5002136, 231.6967926
39: -167.6901855, 77.8157501, -168.2017212, 78.1666260, -245.8568115, 246.0174713
40: -134.9569702, 73.6241455, -135.3500366, 73.8190613, -208.7760162, 208.9741821
41: -100.4363098, 66.9940643, -100.7218170, 67.2615128, -167.6978149, 167.7158661
42: -75.5423584, 65.1448059, -75.8333359, 65.6231995, -141.1655579, 140.9781189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=678, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0411774, upper bound: 99.0643554
time: 117.92 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0524915, upper bound: 99.1201874
time: 102.94 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -124.5078888, 84.2692642, -124.6965485, 84.3848114, -208.8927002, 208.9658203
1: -69.8733978, 74.2441559, -70.0043945, 74.3196869, -144.1930847, 144.2485504
2: -62.5543900, 71.1751862, -62.7342033, 71.3265991, -133.8809662, 133.9093933
3: -72.0444870, 86.1394501, -72.2391205, 86.3287811, -158.3732605, 158.3785706
4: -75.1116333, 84.4709320, -75.3194504, 84.6061401, -159.7177734, 159.7903748
5: -67.3114395, 90.5068817, -67.5017929, 90.7076187, -158.0190582, 158.0086670
6: -102.5066376, 75.5258026, -102.6120377, 75.7159576, -178.2225952, 178.1378326
7: -83.2881622, 91.1501770, -83.4807510, 91.2535248, -174.5416870, 174.6309204
8: -88.3084412, 101.5195160, -88.5274963, 101.7052765, -190.0137177, 190.0470123
9: -78.1861267, 81.4082184, -78.2654724, 81.6197815, -159.8058929, 159.6736908
10: -110.7780457, 117.1133347, -111.0939713, 117.5575485, -228.3355865, 228.2073059
11: -110.6951828, 83.1917725, -110.8665161, 83.3958664, -194.0910492, 194.0582733
12: -110.9210281, 88.2932739, -111.2446365, 88.7043991, -199.6254272, 199.5379028
13: -110.2183456, 100.2125015, -110.2447586, 100.3489685, -210.5673218, 210.4572449
14: -162.6502991, 83.5293732, -162.8938446, 83.7050476, -246.3553314, 246.4232178
15: -91.3251114, 81.4631729, -91.5576935, 81.5627823, -172.8878937, 173.0208740
16: -118.0231781, 96.9976501, -118.1606598, 97.2936630, -215.3168335, 215.1583099
17: -164.2244873, 119.1018982, -164.4307098, 119.3395538, -283.5640259, 283.5325928
18: -101.5612640, 84.5810547, -101.7259140, 84.6604156, -186.2216644, 186.3069763
19: -85.0088730, 47.5413589, -85.1682129, 47.5784760, -132.5873413, 132.7095642
20: -74.6034546, 57.4493027, -74.7317505, 57.4876709, -132.0911255, 132.1810303
21: -104.3750000, 63.0814819, -104.5821304, 63.1489639, -167.5239563, 167.6636047
22: -112.9979401, 72.8536758, -113.1555099, 72.9258804, -185.9238281, 186.0091858
23: -86.3189545, 58.4169884, -86.4245148, 58.4541702, -144.7731018, 144.8414917
24: -103.3800812, 69.3986816, -103.5017090, 69.3427734, -172.7228546, 172.9003906
25: -90.8454361, 68.0320740, -90.9274368, 68.0750580, -158.9205017, 158.9595032
26: -121.9795074, 89.3142242, -122.2082291, 89.5201874, -211.4996948, 211.5224609
27: -104.1169281, 74.1848907, -104.2848663, 74.1015625, -178.2184906, 178.4697571
28: -85.5207062, 63.1420059, -85.6060638, 63.1215782, -148.6422882, 148.7480621
29: -119.1538696, 76.5382233, -119.2648621, 76.6028824, -195.7567444, 195.8030701
30: -102.6486969, 79.5458374, -102.7312393, 79.5862579, -182.2349396, 182.2770691
31: -106.2188034, 67.0111694, -106.3827362, 67.0767975, -173.2955780, 173.3939056
32: -99.7876663, 73.1951370, -99.9350357, 73.3126068, -173.1002808, 173.1301727
33: -140.3958435, 80.5768356, -140.5531769, 80.7219543, -221.1177979, 221.1300049
34: -119.5935745, 72.7066422, -119.7474594, 72.7956390, -192.3892059, 192.4540710
35: -120.0403671, 70.1555557, -120.2335968, 70.2909088, -190.3312378, 190.3891602
36: -117.4573135, 69.5953751, -117.5931168, 69.6639252, -187.1212158, 187.1884918
37: -164.3901367, 73.8309784, -164.5297852, 73.9211731, -238.3113098, 238.3607635
38: -145.2194519, 86.1763000, -145.3960724, 86.2592773, -231.4787140, 231.5723724
39: -167.9221802, 77.8559799, -168.0497131, 77.9503326, -245.8725128, 245.9056854
40: -135.0373230, 73.6391449, -135.1985168, 73.7247620, -208.7620850, 208.8376617
41: -100.4780884, 66.9834137, -100.5640717, 67.1026154, -167.5807037, 167.5474854
42: -75.5519409, 65.0735474, -75.6633759, 65.3003311, -140.8522491, 140.7369232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=679, inp2_unstable=679, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 647

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.9690518, upper bound: 99.0516555
time: 113.55 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.9690518, upper bound: 99.0516555
time: 101.36 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -125.1232986, 84.4939117, -124.9889450, 84.4563751, -209.5796661, 209.4828491
1: -70.3000793, 74.3939056, -70.2091675, 74.3712692, -144.6713562, 144.6030731
2: -63.1959343, 71.3979950, -63.0474319, 71.3743744, -134.5703125, 134.4454346
3: -72.7612305, 86.4390869, -72.5916901, 86.4081573, -159.1693878, 159.0307770
4: -75.8355408, 84.7014999, -75.6701813, 84.6744461, -160.5099487, 160.3716736
5: -67.9681396, 90.8081207, -67.8206482, 90.7801666, -158.7483063, 158.6287537
6: -102.7817078, 75.9368896, -102.7355499, 75.9100189, -178.6917114, 178.6724243
7: -83.8877411, 91.3368073, -83.7641144, 91.3138351, -175.2015686, 175.1009216
8: -89.0284653, 101.8036346, -88.8787689, 101.7732544, -190.8017273, 190.6824036
9: -78.4496384, 81.8926239, -78.3603210, 81.8484955, -160.2981262, 160.2529449
10: -111.3034058, 118.3443985, -111.2326965, 118.1636963, -229.4670715, 229.5770874
11: -111.0321350, 84.1465759, -110.9813156, 83.8728638, -194.9049988, 195.1278687
12: -111.3612289, 89.5854034, -111.3225937, 89.3391266, -200.7003479, 200.9079895
13: -110.5529861, 100.5963974, -110.3982544, 100.5260773, -211.0790710, 210.9946289
14: -163.1486511, 84.2821808, -163.0646973, 84.0721283, -247.2207642, 247.3468628
15: -91.8835297, 81.7258759, -91.8146744, 81.6788177, -173.5623322, 173.5405579
16: -118.4067612, 97.7267609, -118.3215408, 97.6440201, -216.0507507, 216.0483093
17: -164.6124268, 120.2128906, -164.5424500, 119.8830414, -284.4954224, 284.7553406
18: -101.9319153, 85.1652451, -101.8645859, 84.9425125, -186.8744202, 187.0298157
19: -85.3039322, 47.8735733, -85.2613831, 47.7418022, -133.0457306, 133.1349487
20: -74.8834457, 57.7457085, -74.8390732, 57.6336517, -132.5170898, 132.5847778
21: -104.7271118, 63.6089172, -104.6827087, 63.4079933, -168.1351013, 168.2916260
22: -113.3091660, 73.3355255, -113.2876282, 73.1589127, -186.4680786, 186.6231537
23: -86.5514069, 58.7277718, -86.5132904, 58.6039696, -145.1553650, 145.2410583
24: -103.6618805, 69.5074158, -103.6268311, 69.3921204, -173.0540009, 173.1342468
25: -91.0447006, 68.3192902, -91.0089035, 68.2085648, -159.2532501, 159.3281860
26: -122.3781586, 90.2168427, -122.3298874, 89.9580688, -212.3362122, 212.5467224
27: -104.5447540, 74.3089981, -104.4842377, 74.1561890, -178.7009430, 178.7932434
28: -85.7424164, 63.3023415, -85.7000732, 63.1936417, -148.9360657, 149.0024109
29: -119.4038086, 77.1327209, -119.3668365, 76.8939743, -196.2977905, 196.4995575
30: -102.8685837, 79.9680786, -102.8275299, 79.7865524, -182.6551056, 182.7956085
31: -106.5796890, 67.3999100, -106.5220184, 67.2670898, -173.8467712, 173.9219055
32: -100.0851135, 73.6289825, -100.0374146, 73.5227661, -173.6078796, 173.6663818
33: -141.0062866, 80.8697662, -140.8471069, 80.8215179, -221.8277893, 221.7168579
34: -120.0684357, 72.9544678, -119.9746246, 72.8921204, -192.9605560, 192.9290924
35: -120.6278000, 70.3981247, -120.5220871, 70.3638458, -190.9916382, 190.9202118
36: -117.8316345, 69.7797928, -117.7708817, 69.7377548, -187.5693970, 187.5506744
37: -164.7446899, 74.1384125, -164.6845703, 74.0586090, -238.8032990, 238.8229828
38: -145.7759247, 86.3929138, -145.6586914, 86.3418427, -232.1177673, 232.0515747
39: -168.3970337, 78.0620193, -168.2655640, 78.0291519, -246.4261780, 246.3275757
40: -135.4806824, 73.7923813, -135.4015503, 73.7828217, -209.2634888, 209.1939392
41: -100.7368698, 67.2941055, -100.6828613, 67.2470627, -167.9839172, 167.9769592
42: -75.7895660, 65.7809753, -75.7496262, 65.6447449, -141.4342957, 141.5305939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=679, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 647

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.9690518, upper bound: 99.0590815
time: 121.96 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.9690518, upper bound: 99.0590815
time: 109.37 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 233.58 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 233.58
Output dim: 5, lower bound: -99.0411774, upper bound: 98.9449994
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 233.58
Output dim: 5, lower bound: -99.0411774, upper bound: 98.9449994
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 233.58
Output dim: 5, lower bound: -99.0411774, upper bound: 98.9939112
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 233.58
Output dim: 5, lower bound: -99.0411774, upper bound: 99.0573428
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 233.58
Output dim: 5, lower bound: -99.0411774, upper bound: 99.0163707
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 233.58
Output dim: 5, lower bound: -99.0411774, upper bound: 99.0596552
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 233.58
Output dim: 5, lower bound: -99.0411774, upper bound: 99.0643554
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 233.58
Output dim: 5, lower bound: -99.0524915, upper bound: 99.1201874
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 233.58
Output dim: 5, lower bound: -98.9690518, upper bound: 99.0516555
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 233.58
Output dim: 5, lower bound: -98.9690518, upper bound: 99.0516555
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 233.58
Output dim: 5, lower bound: -98.9690518, upper bound: 99.0590815
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 233.58
Output dim: 5, lower bound: -98.9690518, upper bound: 99.0590815
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 233.58
Output dim: 5, lower bound: -99.0170438, upper bound: 99.0569100
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 233.58
Output dim: 5, lower bound: -99.0170438, upper bound: 99.0665439
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=159.03338623046875
rel_dist={5: [-99.13839545242836, 99.13839544583911]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2499324, upper bound: 97.3022695
time: 103.70 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.3024426, upper bound: 97.3024427
time: 163.90 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 267.72 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 267.72
Output dim: 5, lower bound: -97.2499324, upper bound: 97.3022695
IS_A2, status: Status.UNKNOWN, split count: 1, time: 267.72
Output dim: 5, lower bound: -97.3024426, upper bound: 97.3024427

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -124.6066666, 84.2873840, -124.9231720, 84.4195557, -209.0262146, 209.2105408
1: -69.9321289, 74.2248077, -70.1502991, 74.3419724, -144.2741089, 144.3750916
2: -62.7293396, 71.1545944, -63.0056190, 71.3475342, -134.0768738, 134.1602173
3: -72.1667099, 86.0914154, -72.5091705, 86.3658142, -158.5325317, 158.6005707
4: -75.2612152, 84.4494247, -75.5957184, 84.6409149, -159.9021301, 160.0451355
5: -67.5027466, 90.4818497, -67.7862549, 90.7411804, -158.2439270, 158.2680969
6: -102.5512543, 75.7055969, -102.6827545, 75.9060516, -178.4572906, 178.3883514
7: -83.4754562, 91.1614227, -83.7336655, 91.2829437, -174.7583923, 174.8950806
8: -88.5101929, 101.5078583, -88.8101349, 101.7322922, -190.2424927, 190.3179779
9: -78.1663818, 81.6335449, -78.3579788, 81.8004684, -159.9668579, 159.9915161
10: -110.7386780, 117.2646561, -111.1812897, 117.8429260, -228.5815735, 228.4459534
11: -110.5426636, 83.1887283, -110.9406586, 83.7077942, -194.2504578, 194.1293793
12: -110.9254379, 88.7152863, -111.2832489, 89.2170258, -200.1424561, 199.9985352
13: -109.8729401, 100.1326675, -110.2452316, 100.4712067, -210.3441315, 210.3778992
14: -162.6055756, 83.4861450, -163.0047455, 83.9201508, -246.5257111, 246.4908905
15: -91.4326019, 81.5452423, -91.7510071, 81.6459351, -173.0785217, 173.2962341
16: -118.0348358, 97.1803436, -118.2566452, 97.5047455, -215.5395813, 215.4369812
17: -164.1159363, 119.0831833, -164.5087280, 119.7077026, -283.8236389, 283.5919189
18: -101.4866257, 84.3787079, -101.7994003, 84.8154984, -186.3021088, 186.1781006
19: -84.9405899, 47.4515610, -85.2259598, 47.6883698, -132.6289673, 132.6775208
20: -74.5630646, 57.3862228, -74.7972031, 57.5886726, -132.1517334, 132.1834259
21: -104.2998657, 62.9968300, -104.6472626, 63.3420029, -167.6418610, 167.6440735
22: -113.0589981, 72.8634033, -113.2481842, 73.1490555, -186.2080536, 186.1115875
23: -86.2505188, 58.2469292, -86.4806061, 58.5144501, -144.7649689, 144.7275391
24: -103.3558960, 69.1258240, -103.5808640, 69.3379669, -172.6938629, 172.7066956
25: -90.8022842, 67.9390488, -90.9711761, 68.1536255, -158.9559021, 158.9102173
26: -121.9121246, 89.4086609, -122.2773972, 89.8860397, -211.7981262, 211.6860657
27: -104.2366867, 73.9641800, -104.4243317, 74.1748352, -178.4115143, 178.3884888
28: -85.4963226, 63.0298767, -85.6646271, 63.2006683, -148.6969910, 148.6945038
29: -119.1377106, 76.5448532, -119.3299026, 76.9007111, -196.0384064, 195.8747559
30: -102.5404358, 79.3659515, -102.7869873, 79.7212067, -182.2616425, 182.1529388
31: -106.1160583, 66.8169556, -106.4663239, 67.1240997, -173.2401428, 173.2832794
32: -99.8150787, 73.3589783, -99.9853973, 73.5321198, -173.3471985, 173.3443604
33: -140.3820343, 80.5549545, -140.7369232, 80.8017578, -221.1837463, 221.2918701
34: -119.5900650, 72.7058487, -119.8591919, 72.8801727, -192.4702301, 192.5650330
35: -120.0107956, 70.1563034, -120.3416977, 70.3369751, -190.3477478, 190.4980011
36: -117.3122711, 69.6056137, -117.5819626, 69.7298279, -187.0420990, 187.1875610
37: -164.3631897, 73.8990097, -164.5925903, 74.0781708, -238.4413605, 238.4915771
38: -145.1600647, 86.1004486, -145.5035095, 86.3188248, -231.4788666, 231.6039429
39: -167.7764587, 77.8355713, -168.1285095, 78.0059433, -245.7823792, 245.9640656
40: -135.0021057, 73.6946869, -135.2711334, 73.8213196, -208.8234253, 208.9657898
41: -100.4634705, 67.0853043, -100.6291809, 67.2690125, -167.7324829, 167.7144775
42: -75.5640106, 65.2241974, -75.7121582, 65.5519104, -141.1159210, 140.9363251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=679, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.2453988, upper bound: 97.2439578
time: 98.87 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2461684, upper bound: 97.2991105
time: 99.36 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -125.2836914, 84.5387802, -125.3028488, 84.5451355, -209.8288269, 209.8416138
1: -70.4095001, 74.4239502, -70.4218140, 74.4295807, -144.8390808, 144.8457642
2: -63.3581696, 71.4284592, -63.3734131, 71.4330978, -134.7912598, 134.8018799
3: -72.9513702, 86.4834442, -72.9705353, 86.4908981, -159.4422455, 159.4539795
4: -76.0163956, 84.7425385, -76.0337296, 84.7497253, -160.7661133, 160.7762756
5: -68.1348724, 90.8475342, -68.1508789, 90.8534088, -158.9882812, 158.9984131
6: -102.8498077, 76.0811539, -102.8607330, 76.1199646, -178.9697723, 178.9418793
7: -84.0359802, 91.3690567, -84.0512772, 91.3756409, -175.4116211, 175.4203339
8: -89.1975403, 101.8446808, -89.2136230, 101.8518829, -191.0494080, 191.0583038
9: -78.5486755, 81.9712906, -78.5655060, 81.9886551, -160.5373230, 160.5368042
10: -111.3890762, 118.5826645, -111.3990097, 118.6146011, -230.0036774, 229.9816742
11: -111.1012955, 84.4298019, -111.1109695, 84.4556122, -195.5569000, 195.5407715
12: -111.4130249, 89.8627014, -111.4216843, 89.8877258, -201.3007507, 201.2843781
13: -110.7405396, 100.6908951, -110.7526855, 100.7049561, -211.4454498, 211.4435730
14: -163.2510681, 84.5008774, -163.2645569, 84.5219650, -247.7730408, 247.7654266
15: -92.0728302, 81.7929382, -92.1092606, 81.8027191, -173.8755493, 173.9021912
16: -118.5161896, 97.8834991, -118.5300446, 97.9138336, -216.4300232, 216.4135437
17: -164.6907959, 120.5500946, -164.6991882, 120.5821457, -285.2729187, 285.2492676
18: -102.0187836, 85.3914032, -102.0321350, 85.4140320, -187.4328156, 187.4235382
19: -85.3569641, 48.0078011, -85.3636322, 48.0215302, -133.3784943, 133.3714294
20: -74.9406281, 57.8567352, -74.9489288, 57.8669357, -132.8075562, 132.8056641
21: -104.7878342, 63.8062553, -104.7970276, 63.8237000, -168.6115417, 168.6032715
22: -113.3832321, 73.5227127, -113.4052734, 73.5399323, -186.9231567, 186.9279785
23: -86.5999985, 58.8614655, -86.6063385, 58.8755035, -145.4754944, 145.4678040
24: -103.7299271, 69.6165771, -103.7408142, 69.6283722, -173.3582764, 173.3573914
25: -91.0931015, 68.4360886, -91.0994492, 68.4490204, -159.5421143, 159.5355377
26: -122.4458618, 90.4897308, -122.4583588, 90.5140686, -212.9599304, 212.9480591
27: -104.6329498, 74.4486694, -104.6462860, 74.4596634, -179.0926208, 179.0949402
28: -85.7934875, 63.4117355, -85.7997284, 63.4208221, -149.2143097, 149.2114563
29: -119.4589386, 77.3727341, -119.4701767, 77.3934250, -196.8523560, 196.8429108
30: -102.9253006, 80.1761322, -102.9339752, 80.1943054, -183.1195984, 183.1101074
31: -106.6536560, 67.5429840, -106.6637115, 67.5607071, -174.2143555, 174.2066956
32: -100.1512756, 73.7404404, -100.1638184, 73.7507629, -173.9020386, 173.9042664
33: -141.1815643, 80.9286804, -141.2007141, 80.9359589, -222.1175232, 222.1293793
34: -120.1827469, 73.0269165, -120.1975403, 73.0368576, -193.2196045, 193.2244568
35: -120.7642517, 70.4415741, -120.7824707, 70.4469452, -191.2111969, 191.2240295
36: -117.9226379, 69.8307648, -117.9405365, 69.8367462, -187.7593842, 187.7713013
37: -164.8338623, 74.2456055, -164.8507080, 74.2637329, -239.0975952, 239.0962830
38: -145.9229126, 86.4525146, -145.9443970, 86.4589996, -232.3819122, 232.3968964
39: -168.5646667, 78.1052399, -168.5850525, 78.1110611, -246.6757202, 246.6902771
40: -135.5876770, 73.8635330, -135.6048126, 73.8892212, -209.4768982, 209.4683533
41: -100.8066711, 67.4122086, -100.8168564, 67.4406586, -168.2473145, 168.2290649
42: -75.8448944, 65.9455109, -75.8531189, 65.9661331, -141.8110352, 141.7986298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2991863, upper bound: 97.2444423
time: 106.06 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2993029, upper bound: 97.2993028
time: 105.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 213.54 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 213.54
Output dim: 5, lower bound: -97.2453988, upper bound: 97.2439578
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 213.54
Output dim: 5, lower bound: -97.2461684, upper bound: 97.2991105
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 213.54
Output dim: 5, lower bound: -97.2991863, upper bound: 97.2444423
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 213.54
Output dim: 5, lower bound: -97.2993029, upper bound: 97.2993028

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -124.5710602, 84.2745895, -124.9552155, 84.5178833, -209.0889435, 209.2297974
1: -69.9072800, 74.2166748, -70.1608276, 74.4303589, -144.3376312, 144.3775024
2: -62.6941948, 71.1458435, -62.9890480, 71.6002045, -134.2943726, 134.1348877
3: -72.1279907, 86.0779724, -72.4935379, 86.6681366, -158.7961273, 158.5715027
4: -75.2207642, 84.4386139, -75.5842514, 84.8501129, -160.0708771, 160.0228577
5: -67.4688797, 90.4705048, -67.7808914, 91.0635681, -158.5324402, 158.2514038
6: -102.5329971, 75.6209793, -102.7584915, 75.8579407, -178.3909302, 178.3794708
7: -83.4393616, 91.1511993, -83.7436447, 91.3915787, -174.8309326, 174.8948364
8: -88.4745407, 101.4948273, -88.8026123, 101.9582062, -190.4327393, 190.2974396
9: -78.1428146, 81.6031342, -78.3792114, 81.8416595, -159.9844666, 159.9823456
10: -110.7194824, 117.2123718, -111.3396835, 117.8796387, -228.5990906, 228.5520630
11: -110.5166779, 83.1297760, -111.3220367, 83.6639786, -194.1806641, 194.4517822
12: -110.9112396, 88.6579971, -111.7101288, 89.2021179, -200.1133423, 200.3681183
13: -109.7899780, 100.1041183, -110.1856689, 100.6840363, -210.4739990, 210.2897949
14: -162.5788269, 83.4453583, -163.2460632, 83.8934555, -246.4722748, 246.6914062
15: -91.3247223, 81.5236969, -91.7098465, 81.7259979, -173.0507202, 173.2335358
16: -118.0058899, 97.0573425, -118.3787766, 97.4184265, -215.4243164, 215.4361267
17: -164.0960999, 119.0120087, -164.8848724, 119.6484146, -283.7445068, 283.8968811
18: -101.4621048, 84.3336029, -102.0867310, 84.7882233, -186.2503204, 186.4203186
19: -84.9256821, 47.4237404, -85.4938812, 47.6697693, -132.5954285, 132.9176178
20: -74.5458527, 57.3641891, -74.9700012, 57.5821648, -132.1279907, 132.3341675
21: -104.2807236, 62.9589043, -104.9939575, 63.3153992, -167.5961304, 167.9528656
22: -113.0392685, 72.8228760, -113.3653107, 73.1260529, -186.1653137, 186.1881866
23: -86.2385101, 58.2193489, -86.6920319, 58.5085182, -144.7470245, 144.9113617
24: -103.3375320, 69.1065903, -103.7321014, 69.3276749, -172.6651917, 172.8386841
25: -90.7866974, 67.9109650, -91.0502319, 68.1352005, -158.9219055, 158.9611816
26: -121.8899078, 89.3552170, -122.6890259, 89.8638153, -211.7537231, 212.0442505
27: -104.2114258, 73.9385376, -104.6004791, 74.1582565, -178.3696899, 178.5390167
28: -85.4840546, 63.0076218, -85.8641586, 63.1999817, -148.6840363, 148.8717804
29: -119.1194992, 76.4951019, -119.5084305, 76.8540878, -195.9735870, 196.0035095
30: -102.5197144, 79.3264008, -103.0052338, 79.7191086, -182.2388153, 182.3316345
31: -106.0968628, 66.7886047, -106.7466125, 67.1000824, -173.1969452, 173.5352173
32: -99.7960510, 73.3339844, -100.0696945, 73.5495529, -173.3455658, 173.4036865
33: -140.3439331, 80.5380096, -140.7416077, 80.9688263, -221.3127594, 221.2796173
34: -119.5659866, 72.6850891, -119.8932724, 72.9359589, -192.5018921, 192.5783386
35: -119.9745789, 70.1449585, -120.3496399, 70.4248123, -190.3993835, 190.4945984
36: -117.2833710, 69.5919189, -117.6139603, 69.7654037, -187.0487366, 187.2058716
37: -164.3386841, 73.8549957, -164.6949158, 74.0769806, -238.4156647, 238.5499115
38: -145.1240540, 86.0744400, -145.5656128, 86.3668671, -231.4908752, 231.6400452
39: -167.7365112, 77.8232880, -168.1520996, 78.1538849, -245.8903961, 245.9753876
40: -134.9742432, 73.6309433, -135.3065491, 73.8099670, -208.7842102, 208.9375000
41: -100.4468689, 67.0156403, -100.6979294, 67.2435913, -167.6904449, 167.7135620
42: -75.5511017, 65.1887741, -75.8158722, 65.5779800, -141.1290588, 141.0046387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=678, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 647

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.2363991, upper bound: 97.2459027
time: 124.48 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2363991, upper bound: 97.2920153
time: 88.37 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -125.1141052, 84.4901733, -124.9951935, 84.4573975, -209.5715027, 209.4853516
1: -70.2946472, 74.3920898, -70.2139053, 74.3718567, -144.6665039, 144.6059875
2: -63.1819000, 71.3966217, -63.0533943, 71.3754120, -134.5572968, 134.4499969
3: -72.7468872, 86.4387970, -72.5989532, 86.4100266, -159.1569214, 159.0377502
4: -75.8198090, 84.7019577, -75.6767349, 84.6763000, -160.4960938, 160.3786926
5: -67.9568939, 90.8081970, -67.8275681, 90.7821350, -158.7390137, 158.6357727
6: -102.7824173, 75.9712143, -102.7387238, 75.9232101, -178.7056274, 178.7099152
7: -83.8809204, 91.3353653, -83.7706909, 91.3145294, -175.1954498, 175.1060486
8: -89.0169144, 101.8018799, -88.8859100, 101.7743683, -190.7912750, 190.6877899
9: -78.4349518, 81.8955231, -78.3598862, 81.8517761, -160.2867126, 160.2554016
10: -111.2989731, 118.3404465, -111.2361526, 118.1754150, -229.4743958, 229.5765991
11: -111.0313568, 84.1135483, -110.9844742, 83.8801575, -194.9114990, 195.0980225
12: -111.3593979, 89.5677872, -111.3243561, 89.3516083, -200.7109985, 200.8921509
13: -110.5469513, 100.5943146, -110.4092102, 100.5298157, -211.0767670, 211.0035095
14: -163.1429749, 84.2568817, -163.0684204, 84.0779648, -247.2209473, 247.3253021
15: -91.9227371, 81.7261581, -91.8374786, 81.6819458, -173.6046753, 173.5636292
16: -118.4035263, 97.7381210, -118.3262939, 97.6532211, -216.0567474, 216.0643921
17: -164.6058502, 120.1702347, -164.5451355, 119.8911133, -284.4969482, 284.7153625
18: -101.9276810, 85.1343384, -101.8671951, 84.9467545, -186.8744354, 187.0015106
19: -85.3016891, 47.8549500, -85.2635193, 47.7441521, -133.0458374, 133.1184692
20: -74.8811035, 57.7296181, -74.8411255, 57.6360779, -132.5171814, 132.5707397
21: -104.7260361, 63.5795212, -104.6852112, 63.4114990, -168.1375427, 168.2647400
22: -113.3214874, 73.3152313, -113.2924957, 73.1641083, -186.4855652, 186.6077271
23: -86.5498505, 58.7142639, -86.5153961, 58.6081276, -145.1579590, 145.2296600
24: -103.6701736, 69.4866943, -103.6325684, 69.3921661, -173.0623169, 173.1192627
25: -91.0443573, 68.3045502, -91.0113220, 68.2106934, -159.2550507, 159.3158569
26: -122.3761292, 90.1885834, -122.3319550, 89.9667816, -212.3428955, 212.5205383
27: -104.5461884, 74.2823029, -104.4889908, 74.1572189, -178.7033997, 178.7713013
28: -85.7396927, 63.2879257, -85.7020874, 63.1962967, -148.9359741, 148.9900055
29: -119.4029770, 77.1006775, -119.3686218, 76.8993530, -196.3023071, 196.4692841
30: -102.8677216, 79.9584961, -102.8298492, 79.7988129, -182.6665344, 182.7883453
31: -106.5771255, 67.3825760, -106.5250320, 67.2696533, -173.8467712, 173.9076080
32: -100.0826340, 73.6163483, -100.0391846, 73.5267181, -173.6093445, 173.6555176
33: -140.9899750, 80.8666611, -140.8523407, 80.8236465, -221.8136292, 221.7189941
34: -120.0622787, 72.9476471, -119.9788361, 72.8937683, -192.9560242, 192.9264832
35: -120.6220779, 70.3966141, -120.5273209, 70.3656158, -190.9877014, 190.9239349
36: -117.8305054, 69.7768250, -117.7739182, 69.7395782, -187.5700684, 187.5507507
37: -164.7441101, 74.1345749, -164.6879272, 74.0629578, -238.8070679, 238.8225098
38: -145.7678833, 86.3886566, -145.6634827, 86.3439789, -232.1118469, 232.0521393
39: -168.3932495, 78.0610352, -168.2738342, 78.0310364, -246.4242859, 246.3348694
40: -135.4781342, 73.8065262, -135.4056396, 73.7867126, -209.2648468, 209.2121582
41: -100.7342682, 67.3093567, -100.6854630, 67.2548141, -167.9890747, 167.9948120
42: -75.7890167, 65.7726440, -75.7520599, 65.6526184, -141.4416351, 141.5246887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=679, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 647

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.2363991, upper bound: 97.1905344
time: 98.95 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.2363991, upper bound: 97.2371879
time: 123.30 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -125.2459183, 84.5247040, -125.3330994, 84.6412888, -209.8872070, 209.8578033
1: -70.3839035, 74.4152222, -70.4326553, 74.5174332, -144.9013214, 144.8478699
2: -63.3230820, 71.4194946, -63.3586044, 71.6853180, -135.0083923, 134.7780914
3: -72.9119568, 86.4688492, -72.9562302, 86.7913742, -159.7033386, 159.4250793
4: -75.9762421, 84.7303467, -76.0227432, 84.9580002, -160.9342346, 160.7530823
5: -68.1012497, 90.8358383, -68.1467133, 91.1750183, -159.2762756, 158.9825439
6: -102.8296204, 75.9946136, -102.9364624, 76.0669937, -178.8966064, 178.9310608
7: -83.9981995, 91.3582916, -84.0615845, 91.4835739, -175.4817657, 175.4198761
8: -89.1611786, 101.8312225, -89.2066193, 102.0771942, -191.2383728, 191.0378418
9: -78.5261230, 81.9393692, -78.5901794, 82.0285187, -160.5546265, 160.5295410
10: -111.3671341, 118.5322800, -111.5547256, 118.6542206, -230.0213470, 230.0870056
11: -111.0751419, 84.3720856, -111.4890747, 84.4137802, -195.4888916, 195.8611450
12: -111.3974915, 89.8054962, -111.8474426, 89.8730316, -201.2705231, 201.6529388
13: -110.6659241, 100.6597824, -110.7043686, 100.9160614, -211.5819855, 211.3641357
14: -163.2233582, 84.4571381, -163.5048523, 84.4948349, -247.7181702, 247.9619904
15: -91.9708328, 81.7696686, -92.0645752, 81.8819427, -173.8527832, 173.8342438
16: -118.4838409, 97.7605743, -118.6523438, 97.8276367, -216.3114777, 216.4129181
17: -164.6698608, 120.4751282, -165.0742188, 120.5235214, -285.1933899, 285.5493469
18: -101.9913712, 85.3413925, -102.3087463, 85.3871765, -187.3785400, 187.6501160
19: -85.3413544, 47.9797821, -85.6297302, 48.0049171, -133.3462677, 133.6095123
20: -74.9225311, 57.8334045, -75.1205826, 57.8607826, -132.7832947, 132.9539795
21: -104.7679443, 63.7683411, -105.1428833, 63.7985687, -168.5665131, 168.9112244
22: -113.3629532, 73.4780045, -113.5222778, 73.5173035, -186.8802490, 187.0002747
23: -86.5876694, 58.8340645, -86.8172302, 58.8704758, -145.4581451, 145.6512909
24: -103.7103729, 69.5941467, -103.8896408, 69.6181183, -173.3284912, 173.4837952
25: -91.0762939, 68.4050293, -91.1782455, 68.4307861, -159.5070801, 159.5832825
26: -122.4219208, 90.4345551, -122.8676605, 90.4923477, -212.9142761, 213.3022156
27: -104.6061249, 74.4197845, -104.8192520, 74.4430313, -179.0491486, 179.2390442
28: -85.7808838, 63.3880653, -85.9984512, 63.4212036, -149.2020721, 149.3865204
29: -119.4398956, 77.3204193, -119.6475525, 77.3470306, -196.7869263, 196.9679718
30: -102.9041595, 80.1357346, -103.1513062, 80.1935425, -183.0977020, 183.2870483
31: -106.6332245, 67.5119553, -106.9397659, 67.5377350, -174.1709595, 174.4517059
32: -100.1307678, 73.7155151, -100.2464218, 73.7686615, -173.8994293, 173.9619141
33: -141.1427002, 80.9107666, -141.2050781, 81.1011200, -222.2438049, 222.1158447
34: -120.1571274, 73.0043488, -120.2302246, 73.0938568, -193.2509766, 193.2345734
35: -120.7291107, 70.4288330, -120.7918625, 70.5351257, -191.2642365, 191.2207031
36: -117.8932724, 69.8149261, -117.9730225, 69.8727417, -187.7660065, 187.7879486
37: -164.8069763, 74.2012634, -164.9507141, 74.2604523, -239.0674286, 239.1519775
38: -145.8864441, 86.4242477, -146.0065308, 86.5077820, -232.3942261, 232.4307709
39: -168.5246277, 78.0925446, -168.6088409, 78.2578888, -246.7824860, 246.7013855
40: -135.5554047, 73.8028870, -135.6371155, 73.8819504, -209.4373474, 209.4400024
41: -100.7876358, 67.3373642, -100.8833466, 67.4132156, -168.2008514, 168.2207031
42: -75.8291702, 65.9068069, -75.9548492, 65.9899445, -141.8191223, 141.8616638

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.1985144, upper bound: 97.2291158
time: 123.84 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.1985144, upper bound: 97.2944053
time: 103.38 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 229.41 seconds
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 229.41
Output dim: 5, lower bound: -97.2363991, upper bound: 97.2459027
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 229.41
Output dim: 5, lower bound: -97.2363991, upper bound: 97.2920153
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 229.41
Output dim: 5, lower bound: -97.2363991, upper bound: 97.1905344
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 229.41
Output dim: 5, lower bound: -97.2363991, upper bound: 97.2371879
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 229.41
Output dim: 5, lower bound: -97.1985144, upper bound: 97.2291158
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 229.41
Output dim: 5, lower bound: -97.1985144, upper bound: 97.2944053

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -124.5469818, 84.2570038, -124.9419174, 84.5080414, -209.0550232, 209.1989136
1: -69.8930664, 74.2049484, -70.1529236, 74.4239960, -144.3170624, 144.3578796
2: -62.6797905, 71.1357727, -62.9810333, 71.5947342, -134.2745056, 134.1168060
3: -72.1145554, 86.0621033, -72.4861298, 86.6595459, -158.7741089, 158.5482178
4: -75.2056732, 84.4283524, -75.5758209, 84.8445053, -160.0501709, 160.0041504
5: -67.4567413, 90.4546967, -67.7740936, 91.0550385, -158.5117798, 158.2287903
6: -102.5179901, 75.5680923, -102.7502213, 75.8294601, -178.3474426, 178.3183136
7: -83.4220428, 91.1353149, -83.7340240, 91.3830719, -174.8050842, 174.8693390
8: -88.4596024, 101.4829865, -88.7943039, 101.9517136, -190.4113159, 190.2772827
9: -78.1282959, 81.5927887, -78.3700256, 81.8358154, -159.9641113, 159.9627991
10: -110.6994629, 117.1922073, -111.3286057, 117.8683777, -228.5678253, 228.5208130
11: -110.4944229, 83.0338593, -111.3096619, 83.6124649, -194.1068878, 194.3435211
12: -110.8888550, 88.6424637, -111.6981201, 89.1935120, -200.0823669, 200.3405762
13: -109.7485657, 100.0821228, -110.1578217, 100.6718826, -210.4204407, 210.2399292
14: -162.5534363, 83.4390793, -163.2324219, 83.8899612, -246.4433899, 246.6714935
15: -91.2629089, 81.5084229, -91.6765518, 81.7174835, -172.9803925, 173.1849670
16: -117.9810257, 96.9664841, -118.3649216, 97.3708344, -215.3518372, 215.3314056
17: -164.0793457, 118.9882660, -164.8758698, 119.6354523, -283.7147827, 283.8641357
18: -101.4468231, 84.3203278, -102.0777130, 84.7808533, -186.2276611, 186.3980255
19: -84.9157104, 47.4134827, -85.4883270, 47.6638908, -132.5796051, 132.9018097
20: -74.5340881, 57.3544960, -74.9636383, 57.5768089, -132.1108704, 132.3181305
21: -104.2659302, 62.9406776, -104.9859314, 63.3051605, -167.5710907, 167.9265900
22: -112.9924774, 72.8032761, -113.3400879, 73.1150970, -186.1075745, 186.1433563
23: -86.2294235, 58.1861877, -86.6869965, 58.4904633, -144.7198792, 144.8731842
24: -103.3204193, 69.0979919, -103.7226181, 69.3229370, -172.6433563, 172.8206024
25: -90.7684860, 67.8982697, -91.0398254, 68.1281586, -158.8966370, 158.9380798
26: -121.8653717, 89.3414307, -122.6759186, 89.8560715, -211.7214050, 212.0173492
27: -104.1924362, 73.9211121, -104.5901718, 74.1467590, -178.3392029, 178.5112610
28: -85.4767761, 62.9942245, -85.8601837, 63.1925659, -148.6693115, 148.8544006
29: -119.0918579, 76.4772491, -119.4933167, 76.8440857, -195.9359131, 195.9705658
30: -102.5026321, 79.2823257, -102.9958954, 79.6951981, -182.1978302, 182.2781982
31: -106.0833740, 66.7448273, -106.7387238, 67.0764771, -173.1598358, 173.4835510
32: -99.7791748, 73.3200836, -100.0606079, 73.5418396, -173.3210144, 173.3806763
33: -140.3286743, 80.5241623, -140.7332001, 80.9612732, -221.2899475, 221.2573242
34: -119.5551300, 72.6664734, -119.8873062, 72.9256058, -192.4807434, 192.5537720
35: -119.9594421, 70.1327820, -120.3413086, 70.4181519, -190.3775940, 190.4740906
36: -117.2538757, 69.5817413, -117.5979309, 69.7596436, -187.0135193, 187.1796722
37: -164.3092651, 73.8449554, -164.6787109, 74.0713272, -238.3805847, 238.5236664
38: -145.1102448, 86.0629730, -145.5578613, 86.3605881, -231.4708252, 231.6208344
39: -167.6842041, 77.8139801, -168.1237183, 78.1488342, -245.8330078, 245.9376984
40: -134.9530640, 73.6151581, -135.2948303, 73.8013992, -208.7544403, 208.9099884
41: -100.4338760, 66.9835129, -100.6907959, 67.2262192, -167.6600952, 167.6742859
42: -75.5404892, 65.1395264, -75.8099823, 65.5509491, -141.0914307, 140.9495087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=678, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1784137
time: 89.62 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.2194675, upper bound: 97.2371882
time: 118.56 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -125.2182770, 84.5180511, -125.3179321, 84.6376877, -209.8559570, 209.8359680
1: -70.3644104, 74.4101181, -70.4218750, 74.5146484, -144.8790588, 144.8320007
2: -63.2988281, 71.4140015, -63.3453217, 71.6823273, -134.9811554, 134.7593231
3: -72.8818741, 86.4594727, -72.9399185, 86.7862701, -159.6681519, 159.3993835
4: -75.9492340, 84.7211227, -76.0079346, 84.9529800, -160.9022217, 160.7290497
5: -68.0738068, 90.8272400, -68.1318054, 91.1703796, -159.2441864, 158.9590454
6: -102.8143158, 75.9370041, -102.9281006, 76.0364838, -178.8507996, 178.8651123
7: -83.9713135, 91.3524399, -84.0469284, 91.4803314, -175.4516296, 175.3993683
8: -89.1334915, 101.8237610, -89.1914673, 102.0731888, -191.2066650, 191.0152283
9: -78.5163116, 81.9202118, -78.5847778, 82.0180817, -160.5343933, 160.5049896
10: -111.3520355, 118.4836502, -111.5464783, 118.6275177, -229.9795380, 230.0301208
11: -111.0607910, 84.3363342, -111.4814453, 84.3943176, -195.4550934, 195.8177795
12: -111.3877258, 89.7588959, -111.8421326, 89.8477783, -201.2354736, 201.6010284
13: -110.6346512, 100.6408920, -110.6875076, 100.9057388, -211.5403900, 211.3283997
14: -163.2055511, 84.4293518, -163.4951172, 84.4799347, -247.6854858, 247.9244690
15: -91.8995361, 81.7549057, -92.0220642, 81.8739014, -173.7734070, 173.7769775
16: -118.4626846, 97.7186966, -118.6407166, 97.8049011, -216.2675629, 216.3594055
17: -164.6579590, 120.4352417, -165.0677032, 120.5018997, -285.1598511, 285.5029297
18: -101.9758377, 85.3165359, -102.3003464, 85.3738098, -187.3496399, 187.6168823
19: -85.3316040, 47.9652710, -85.6244583, 47.9969139, -133.3285065, 133.5897217
20: -74.9119110, 57.8219032, -75.1148834, 57.8540039, -132.7658997, 132.9367828
21: -104.7556152, 63.7484550, -105.1362152, 63.7877502, -168.5433350, 168.8846741
22: -113.3371124, 73.4533081, -113.5084991, 73.5037537, -186.8408661, 186.9617920
23: -86.5783234, 58.8156052, -86.8121948, 58.8602600, -145.4385834, 145.6278076
24: -103.6890488, 69.5866394, -103.8782883, 69.6140366, -173.3030853, 173.4649353
25: -91.0660248, 68.3912277, -91.1727753, 68.4231949, -159.4892273, 159.5639954
26: -122.4087982, 90.3973465, -122.8605042, 90.4722290, -212.8810272, 213.2578430
27: -104.5858002, 74.4103699, -104.8081665, 74.4378967, -179.0236969, 179.2185364
28: -85.7719116, 63.3756142, -85.9935608, 63.4145317, -149.1864471, 149.3691711
29: -119.4285965, 77.2933350, -119.6412659, 77.3322372, -196.7608337, 196.9346008
30: -102.8924942, 80.0980225, -103.1450195, 80.1711426, -183.0636292, 183.2430267
31: -106.6191864, 67.4944916, -106.9322357, 67.5281525, -174.1473389, 174.4267273
32: -100.1184311, 73.7012177, -100.2397003, 73.7602921, -173.8787231, 173.9409180
33: -141.1174927, 80.9003906, -141.1915588, 81.0955505, -222.2130432, 222.0919189
34: -120.1371307, 72.9940186, -120.2194595, 73.0881653, -193.2252808, 193.2134705
35: -120.7045670, 70.4205933, -120.7785797, 70.5306091, -191.2351685, 191.1991730
36: -117.8746719, 69.8061829, -117.9628220, 69.8678970, -187.7425690, 187.7690125
37: -164.7880402, 74.1809845, -164.9403992, 74.2477417, -239.0357819, 239.1213837
38: -145.8610992, 86.4147339, -145.9926147, 86.5026245, -232.3636780, 232.4073486
39: -168.4913483, 78.0839233, -168.5909576, 78.2532578, -246.7445984, 246.6748810
40: -135.5341492, 73.7763214, -135.6254883, 73.8674545, -209.4015961, 209.4017944
41: -100.7745361, 67.2996902, -100.8761978, 67.3928909, -168.1674194, 168.1758881
42: -75.8176575, 65.8778076, -75.9486389, 65.9727478, -141.7904053, 141.8264313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 647

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.1621311, upper bound: 97.2888398
time: 110.17 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.1910976, upper bound: 97.2888398
time: 118.11 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 230.46 seconds
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 230.46
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1784137
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 230.46
Output dim: 5, lower bound: -97.2194675, upper bound: 97.2371882
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 230.46
Output dim: 5, lower bound: -97.1621311, upper bound: 97.2888398
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 230.46
Output dim: 5, lower bound: -97.1910976, upper bound: 97.2888398

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -125.1571426, 84.3450851, -125.0675430, 84.3058929, -209.4630127, 209.4126282
1: -70.3279114, 74.2947388, -70.2414551, 74.2884216, -144.6163330, 144.5361938
2: -63.2639160, 71.2759857, -63.1404037, 71.4269867, -134.6909027, 134.4163818
3: -72.8487396, 86.2680664, -72.6914520, 86.4291229, -159.2778625, 158.9595184
4: -75.9079666, 84.6690063, -75.8359222, 84.8380737, -160.7460327, 160.5049133
5: -68.0318909, 90.6141205, -67.8549042, 90.7718964, -158.8037720, 158.4690247
6: -102.7644348, 75.7598724, -102.7740326, 75.6749191, -178.4393616, 178.5339050
7: -83.9047241, 91.0896378, -83.6821594, 90.9974518, -174.9021759, 174.7717896
8: -89.1026459, 101.6797943, -88.9809570, 101.7996368, -190.9022675, 190.6607513
9: -78.4045486, 81.8824768, -78.3532333, 81.8155060, -160.2200623, 160.2357025
10: -111.1597519, 118.4343033, -111.1674728, 118.2853317, -229.4450836, 229.6017609
11: -110.9748993, 84.2929993, -111.3428726, 84.1819763, -195.1568604, 195.6358643
12: -111.0197220, 89.7116394, -111.1788177, 89.3876038, -200.4073181, 200.8904419
13: -110.4750366, 100.5548477, -110.3581085, 100.6096573, -211.0846863, 210.9129486
14: -162.8968506, 84.4005661, -162.8806763, 84.1513672, -247.0481873, 247.2812500
15: -91.6909485, 81.6968918, -91.5505219, 81.6256485, -173.3165894, 173.2473907
16: -118.3731308, 97.6076508, -118.3814087, 97.5031052, -215.8762054, 215.9890442
17: -164.4254150, 120.3910980, -164.6144409, 120.1647491, -284.5901489, 285.0055237
18: -101.8664246, 85.2823563, -102.0606537, 85.2495117, -187.1159363, 187.3430176
19: -85.2727890, 47.9462090, -85.5060349, 47.9379158, -133.2107086, 133.4522400
20: -74.8350830, 57.7969322, -74.9340668, 57.7687798, -132.6038666, 132.7309875
21: -104.6744156, 63.7183266, -104.9935913, 63.6723289, -168.3467255, 168.7118988
22: -113.0587692, 73.3972626, -112.9659424, 73.1759262, -186.2346802, 186.3632050
23: -86.5201950, 58.7764931, -86.6828613, 58.7556496, -145.2758484, 145.4593506
24: -103.6295090, 69.5555344, -103.7226486, 69.5343170, -173.1638184, 173.2781830
25: -90.9525757, 68.3413849, -90.9426498, 68.2405548, -159.1931305, 159.2840271
26: -122.0297546, 90.3545685, -122.1394501, 90.0701904, -212.0999146, 212.4940186
27: -104.5237961, 74.3372879, -104.5725784, 74.2837372, -178.8075256, 178.9098663
28: -85.7240448, 63.3323135, -85.8572159, 63.3115044, -149.0355530, 149.1895294
29: -119.2341766, 77.2447205, -119.2629623, 77.0133896, -196.2475586, 196.5076752
30: -102.8323441, 80.0112152, -102.9923706, 79.9643860, -182.7967224, 183.0035858
31: -106.5490570, 67.4549408, -106.7900162, 67.4149780, -173.9640350, 174.2449646
32: -100.0287476, 73.6567535, -100.0461578, 73.6034698, -173.6322174, 173.7029114
33: -141.0638123, 80.8340149, -140.9709320, 80.9614029, -222.0252075, 221.8049469
34: -120.0806503, 72.9071045, -119.9808273, 72.9068146, -192.9874573, 192.8879089
35: -120.6570435, 70.3504257, -120.5820618, 70.3878479, -191.0448914, 190.9324646
36: -117.8066406, 69.7525330, -117.7821579, 69.7615128, -187.5681458, 187.5346985
37: -164.6941223, 74.1434250, -164.7020111, 74.1417160, -238.8358459, 238.8454285
38: -145.8062134, 86.3541107, -145.7745667, 86.3721237, -232.1783447, 232.1286774
39: -168.4046631, 78.0264435, -168.3581543, 78.1398468, -246.5444641, 246.3845978
40: -135.4767151, 73.6490479, -135.4093018, 73.6279526, -209.1046753, 209.0583496
41: -100.7287216, 67.1800232, -100.7289047, 67.1390152, -167.8677368, 167.9089050
42: -75.7717438, 65.8210144, -75.8737183, 65.7689972, -141.5407410, 141.6947021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=501, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=679, inp2_unstable=679, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.1157873, upper bound: 97.2834341
time: 129.17 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.1157873, upper bound: 97.2848615
time: 108.08 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -125.2035141, 84.5077515, -125.2909546, 84.6191864, -209.8226929, 209.7987061
1: -70.3554230, 74.4033966, -70.4054871, 74.5022049, -144.8576355, 144.8088684
2: -63.2905083, 71.4081802, -63.3300323, 71.6716309, -134.9621429, 134.7382050
3: -72.8740387, 86.4501801, -72.9256897, 86.7693024, -159.6433411, 159.3758698
4: -75.9402313, 84.7146606, -75.9914780, 84.9412842, -160.8815155, 160.7061462
5: -68.0663376, 90.8182678, -68.1182785, 91.1539154, -159.2202454, 158.9365234
6: -102.8039246, 75.9078674, -102.9091721, 75.9815140, -178.7854309, 178.8170319
7: -83.9590225, 91.3434753, -84.0244751, 91.4637299, -175.4227295, 175.3679504
8: -89.1247330, 101.8166809, -89.1753845, 102.0601120, -191.1848297, 190.9920654
9: -78.5049133, 81.9139099, -78.5652161, 82.0065613, -160.5114746, 160.4791260
10: -111.3395081, 118.4713211, -111.5237885, 118.6053009, -229.9448090, 229.9951172
11: -111.0474243, 84.2820282, -111.4566040, 84.2934418, -195.3408661, 195.7386169
12: -111.3750458, 89.7498627, -111.8187561, 89.8311768, -201.2061768, 201.5686188
13: -110.5983200, 100.6263046, -110.6230011, 100.8792191, -211.4775085, 211.2492981
14: -163.1911011, 84.4249649, -163.4685822, 84.4719543, -247.6630554, 247.8935394
15: -91.8590088, 81.7439346, -91.9569778, 81.8540344, -173.7130280, 173.7009125
16: -118.4462509, 97.6677933, -118.6106491, 97.7092667, -216.1555023, 216.2784271
17: -164.6484222, 120.4215393, -165.0501404, 120.4764938, -285.1249084, 285.4716797
18: -101.9649963, 85.3087234, -102.2807312, 85.3593826, -187.3243713, 187.5894318
19: -85.3252716, 47.9590607, -85.6130371, 47.9855194, -133.3107910, 133.5720978
20: -74.9047699, 57.8162041, -75.1018219, 57.8435326, -132.7483063, 132.9180145
21: -104.7463150, 63.7382355, -105.1195374, 63.7684479, -168.5147552, 168.8577728
22: -113.3111954, 73.4390030, -113.4601440, 73.4776993, -186.7888947, 186.8991394
23: -86.5727997, 58.7974701, -86.8022614, 58.8264236, -145.3992310, 145.5997314
24: -103.6788101, 69.5815125, -103.8597107, 69.6047592, -173.2835693, 173.4412231
25: -91.0545959, 68.3825378, -91.1535873, 68.4073334, -159.4619293, 159.5361328
26: -122.3947220, 90.3884430, -122.8346405, 90.4558563, -212.8505859, 213.2230835
27: -104.5739594, 74.3988953, -104.7866898, 74.4165039, -178.9904633, 179.1855774
28: -85.7675476, 63.3674355, -85.9856262, 63.4001961, -149.1677399, 149.3530579
29: -119.4129105, 77.2811508, -119.6123657, 77.3100891, -196.7229767, 196.8934937
30: -102.8821564, 80.0745850, -103.1260452, 80.1256561, -183.0077972, 183.2006226
31: -106.6103745, 67.4702072, -106.9162292, 67.4827347, -174.0931091, 174.3864441
32: -100.1083298, 73.6931458, -100.2213211, 73.7454758, -173.8537903, 173.9144592
33: -141.1083984, 80.8915253, -141.1750793, 81.0795517, -222.1879578, 222.0666046
34: -120.1305542, 72.9830246, -120.2076111, 73.0683594, -193.1989136, 193.1906281
35: -120.6955414, 70.4127731, -120.7621536, 70.5163116, -191.2118225, 191.1749268
36: -117.8582077, 69.7988205, -117.9320526, 69.8545456, -187.7127533, 187.7308655
37: -164.7707520, 74.1750031, -164.9087677, 74.2367935, -239.0075378, 239.0837708
38: -145.8524933, 86.4079361, -145.9769440, 86.4902191, -232.3426971, 232.3848724
39: -168.4623413, 78.0781250, -168.5367432, 78.2427979, -246.7051239, 246.6148682
40: -135.5204163, 73.7666473, -135.6002655, 73.8502884, -209.3706970, 209.3668823
41: -100.7658691, 67.2819824, -100.8604202, 67.3601074, -168.1259766, 168.1423950
42: -75.8103714, 65.8495255, -75.9355240, 65.9193726, -141.7297363, 141.7850494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=501, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.1447302, upper bound: 97.2834341
time: 107.07 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2848612, upper bound: 97.2848615
time: 116.19 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 225.49 seconds
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 225.49
Output dim: 5, lower bound: -97.1157873, upper bound: 97.2834341
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 225.49
Output dim: 5, lower bound: -97.1157873, upper bound: 97.2848615
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 225.49
Output dim: 5, lower bound: -97.1447302, upper bound: 97.2834341
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 225.49
Output dim: 5, lower bound: -97.2848612, upper bound: 97.2848615

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -125.1696091, 84.4441223, -125.0137253, 84.2824783, -209.4520874, 209.4578552
1: -70.3348160, 74.3354645, -70.2067719, 74.2722549, -144.6070709, 144.5422363
2: -63.2411652, 71.4248123, -63.0958290, 71.4116211, -134.6527863, 134.5206299
3: -72.8401566, 86.4646912, -72.6417847, 86.4056091, -159.2457428, 159.1064758
4: -75.8769989, 84.8304291, -75.7801666, 84.8184204, -160.6954193, 160.6105957
5: -68.0351028, 90.8454819, -67.8140030, 90.7510681, -158.7861633, 158.6594543
6: -102.8293762, 75.6944275, -102.7432251, 75.5746765, -178.4040527, 178.4376373
7: -83.9132538, 91.0966339, -83.6351929, 90.9537964, -174.8670349, 174.7318115
8: -89.0813828, 101.7559586, -88.9289322, 101.7788620, -190.8602448, 190.6848907
9: -78.3913651, 81.9653244, -78.3067932, 81.7863007, -160.1776733, 160.2720947
10: -111.1936722, 118.4583054, -111.1187744, 118.2068558, -229.4005280, 229.5770874
11: -111.3539886, 84.2280579, -111.3037109, 84.0971375, -195.4511261, 195.5317688
12: -111.3348923, 89.6797485, -111.1524277, 89.3144836, -200.6493683, 200.8321838
13: -110.3679886, 100.7004013, -110.2376480, 100.5658188, -210.9338074, 210.9380493
14: -163.1516418, 84.3537750, -162.8382568, 84.0921478, -247.2437744, 247.1920319
15: -91.5977631, 81.7790680, -91.4032593, 81.5950775, -173.1928406, 173.1823120
16: -118.4290161, 97.5161133, -118.3298950, 97.3538818, -215.7828979, 215.8460083
17: -164.9385071, 120.2898254, -164.5805817, 120.0492706, -284.9877625, 284.8703918
18: -102.0489807, 85.2433929, -102.0227432, 85.1793976, -187.2283783, 187.2661133
19: -85.5167542, 47.9239960, -85.4816895, 47.8955650, -133.4123230, 133.4056854
20: -75.0111542, 57.7792053, -74.9072266, 57.7318802, -132.7430420, 132.6864319
21: -105.0252838, 63.6751823, -104.9638062, 63.6160164, -168.6412964, 168.6389923
22: -113.2059250, 73.3626709, -112.9361420, 73.1143188, -186.3202515, 186.2987976
23: -86.7097778, 58.7510262, -86.6625671, 58.7161179, -145.4258728, 145.4135895
24: -103.7417679, 69.5501404, -103.6915436, 69.5025864, -173.2443237, 173.2416687
25: -91.0252686, 68.3165131, -90.9079437, 68.1964340, -159.2216797, 159.2244568
26: -122.4029999, 90.3082428, -122.1033936, 90.0014496, -212.4044495, 212.4116211
27: -104.7183228, 74.3157806, -104.5337830, 74.2397461, -178.9580688, 178.8495636
28: -85.9348755, 63.3161888, -85.8378601, 63.2721443, -149.2070160, 149.1540527
29: -119.4769058, 77.1696472, -119.2345428, 76.9365540, -196.4134521, 196.4041748
30: -103.0632858, 79.9883423, -102.9586868, 79.9033508, -182.9666138, 182.9469910
31: -106.7848587, 67.4242706, -106.7585373, 67.3716583, -174.1565247, 174.1828003
32: -100.0806961, 73.6496277, -100.0150146, 73.5463867, -173.6270599, 173.6646271
33: -141.0529785, 80.9845581, -140.9067078, 80.9337387, -221.9867249, 221.8912659
34: -120.0808563, 72.9764481, -119.9312897, 72.8801270, -192.9609680, 192.9077454
35: -120.6553421, 70.4174347, -120.5308075, 70.3677750, -191.0231171, 190.9482117
36: -117.8252640, 69.7797089, -117.7348404, 69.7287827, -187.5540466, 187.5145416
37: -164.7504883, 74.1425705, -164.6531067, 74.0919647, -238.8424377, 238.7956696
38: -145.8566895, 86.4211807, -145.7166443, 86.3415222, -232.1982117, 232.1377869
39: -168.3841858, 78.2018356, -168.2896881, 78.1210556, -246.5052185, 246.4915009
40: -135.4736938, 73.6759949, -135.3508301, 73.5722351, -209.0459290, 209.0268250
41: -100.7600784, 67.1407700, -100.6944275, 67.0534821, -167.8135529, 167.8352051
42: -75.8375549, 65.8146820, -75.8483429, 65.7145309, -141.5520935, 141.6630249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=500, inp2_unstable=501, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1672

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.0958926, upper bound: 97.2150209
time: 126.28 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.2313887, upper bound: 97.2819819
time: 505.83 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -125.2152634, 84.6065979, -125.2370148, 84.5958099, -209.8110657, 209.8436127
1: -70.3619385, 74.4440231, -70.3707886, 74.4860229, -144.8479614, 144.8148041
2: -63.2674866, 71.5568390, -63.2853928, 71.6562119, -134.9237061, 134.8422241
3: -72.8652649, 86.6463928, -72.8759918, 86.7456665, -159.6109314, 159.5223694
4: -75.9090424, 84.8757935, -75.9357452, 84.9216537, -160.8306885, 160.8115387
5: -68.0691986, 91.0494461, -68.0773087, 91.1330109, -159.2021790, 159.1267548
6: -102.8678207, 75.8425293, -102.8783112, 75.8812256, -178.7490540, 178.7208405
7: -83.9662781, 91.3506012, -83.9775925, 91.4201126, -175.3863831, 175.3281860
8: -89.1032562, 101.8926086, -89.1233597, 102.0393066, -191.1425476, 191.0159302
9: -78.4917145, 81.9966660, -78.5189514, 81.9772491, -160.4689636, 160.5156250
10: -111.3725586, 118.4949188, -111.4750900, 118.5267715, -229.8993225, 229.9699707
11: -111.4261017, 84.2164154, -111.4172897, 84.2085876, -195.6346893, 195.6336975
12: -111.6899567, 89.7175751, -111.7922287, 89.7579346, -201.4478912, 201.5097961
13: -110.4903717, 100.7706680, -110.5026321, 100.8355026, -211.3258667, 211.2733002
14: -163.4458466, 84.3778152, -163.4260559, 84.4127579, -247.8585815, 247.8038025
15: -91.7665253, 81.8249359, -91.8103027, 81.8232117, -173.5897369, 173.6352234
16: -118.5007401, 97.5762024, -118.5589981, 97.5601654, -216.0609131, 216.1351929
17: -165.1612549, 120.3198318, -165.0161591, 120.3610458, -285.5222473, 285.3359985
18: -102.1470413, 85.2692642, -102.2426605, 85.2892303, -187.4362793, 187.5119324
19: -85.5685654, 47.9366570, -85.5886078, 47.9431190, -133.5116882, 133.5252686
20: -75.0804443, 57.7982674, -75.0749664, 57.8065834, -132.8870239, 132.8732300
21: -105.0963440, 63.6947060, -105.0896912, 63.7120934, -168.8084259, 168.7843933
22: -113.4583588, 73.4029465, -113.4302673, 73.4159698, -186.8743286, 186.8332214
23: -86.7621765, 58.7719231, -86.7818909, 58.7868690, -145.5490417, 145.5538025
24: -103.7906876, 69.5759964, -103.8285217, 69.5730286, -173.3637085, 173.4044800
25: -91.1272812, 68.3570404, -91.1189270, 68.3632050, -159.4904785, 159.4759674
26: -122.7676086, 90.3412552, -122.7984848, 90.3870316, -213.1546173, 213.1397400
27: -104.7685471, 74.3771286, -104.7478943, 74.3724670, -179.1410217, 179.1250153
28: -85.9780884, 63.3512421, -85.9662018, 63.3608170, -149.3388977, 149.3174438
29: -119.6553116, 77.2052307, -119.5838852, 77.2332840, -196.8885651, 196.7891235
30: -103.1125488, 80.0514603, -103.0922699, 80.0646286, -183.1771851, 183.1437378
31: -106.8461456, 67.4392242, -106.8845673, 67.4394073, -174.2855530, 174.3237762
32: -100.1599121, 73.6858063, -100.1901321, 73.6884232, -173.8483276, 173.8759308
33: -141.0972443, 81.0416641, -141.1107788, 81.0517883, -222.1490326, 222.1524353
34: -120.1305466, 73.0521088, -120.1580505, 73.0417480, -193.1723022, 193.2101440
35: -120.6934967, 70.4789581, -120.7108154, 70.4960556, -191.1895447, 191.1897736
36: -117.8763580, 69.8249893, -117.8847733, 69.8217010, -187.6980591, 187.7097626
37: -164.8265228, 74.1737747, -164.8598022, 74.1868591, -239.0133820, 239.0335693
38: -145.9024658, 86.4746704, -145.9189148, 86.4595184, -232.3619843, 232.3935852
39: -168.4412994, 78.2531738, -168.4681854, 78.2238007, -246.6651001, 246.7213593
40: -135.5165100, 73.7942047, -135.5417786, 73.7948532, -209.3113403, 209.3359680
41: -100.7965393, 67.2433777, -100.8259277, 67.2748718, -168.0714111, 168.0693054
42: -75.8753815, 65.8422928, -75.9099197, 65.8648987, -141.7402802, 141.7521973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=500, inp2_unstable=501, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1672

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.2737413, upper bound: 97.2150209
time: 110.94 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.2819818, upper bound: 97.2819819
time: 163.40 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 276.57 seconds
IS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 276.57
Output dim: 5, lower bound: -97.0958926, upper bound: 97.2150209
IS_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 276.57
Output dim: 5, lower bound: -97.2313887, upper bound: 97.2819819
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 276.57
Output dim: 5, lower bound: -97.2737413, upper bound: 97.2150209
IS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 276.57
Output dim: 5, lower bound: -97.2819818, upper bound: 97.2819819
Binary search (step 2): status=Status.VERIFIED, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=159.03338623046875
rel_dist={5: [-97.30393826817422, 97.30393825827855]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 11325.25 seconds
