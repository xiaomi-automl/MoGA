# MoGA: Searching Beyond MobileNetV3

We propose the first Mobile GPU-Aware (MoGA) neural architecture search, so to be precisely tailored for real-world applications. Further, the ultimate objective to devise a mobile network lies in achieving better performance by maximizing the utilization of bounded resources. Counterintuitively, while urging higher capability and restraining their time consumption, we step forward to find networks by enlarging their numbers of parameters as many as possible to increase their representational power. We deliver our searched networks at a mobile scale that outperform MobileNetV3 under the same latency constraints, i.e., MoGA-A achieves 75.9\% top-1 accuracy on ImageNet, with the same CPU latency as MobileNetV3 which scores 75.2\% , MoGA-B meets 75.4\% with same mobile GPU latency.

## MoGA Architectures
![](images/moga_arch.png)

## Requirements
* Python 3.6 +
* Pytorch 1.0.1 +
* The pretrained models are accessible after submitting a questionnaire: https://forms.gle/o2cUfQPieVcm3t8B8.

## Benchmarks on ImageNet

![](images/specs.png)


## ImageNet Dataset

We use the standard ImageNet 2012 dataset, the only difference is that we reorganized the validation set by their classes. 
