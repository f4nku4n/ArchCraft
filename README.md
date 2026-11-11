<div align="center">
<h1> Revisiting Neural Networks for Continual Learning:<br />
  An Architectural Perspective
</h1>
<div>
    <a>Aojun Lu</a>&emsp;
    <a target='_blank'>Tao Feng</a>&emsp;
    <a href='https://jacobyuan7.github.io/' target='_blank'>Hangjie Yuan</a>&emsp;
    <a>Xiaotian Song</a>&emsp;
    <a href='https://yn-sun.github.io/' target='_blank'>Yanan Sun&#9993</a>&emsp;
</div>

<strong>Accepted to <a href='https://ijcai24.org/' target='_blank'>IJCAI 2024</a> :partying_face:</strong>

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2404.07965)
[![GitHub Stars](https://img.shields.io/github/stars/byyx666/ArchCraft?style=social)](https://github.com/byyx666/ArchCraft)
[![GitHub Forks](https://img.shields.io/github/forks/byyx666/ArchCraft)](https://github.com/byyx666/ArchCraft)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fbyyx666%2FArchCraft&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
</div>

![colored_mesh (1)](teaser.png)

> Abstract:
> Efforts to overcome catastrophic forgetting have primarily centered around developing more effective Continual Learning (CL) methods. In contrast, less attention was devoted to analyzing the role of network architecture design (e.g., network depth, width, and components) in contributing to CL. This paper seeks to bridge this gap between network architecture design and CL, and to present a holistic study on the impact of network architectures on CL. This work considers architecture design at the network scaling level, i.e., width and depth, and also at the network components, i.e., skip connections, global pooling layers, and down-sampling. In both cases, we first derive insights through systematically exploring how architectural designs affect CL. Then, grounded in these insights, we craft a specialized search space for CL and further propose a simple yet effective ArchCraft method to steer a CL-friendly architecture, namely, this method recrafts AlexNet/ResNet into AlexAC/ResAC. Experimental validation across various CL settings and scenarios demonstrates that improved architectures are parameter-efficient, achieving state-of-the-art performance of CL while being 86%, 61%, and 97% more compact in terms of parameters than the naive CL architecture in Task IL and Class IL. 

## Information before using this repo
ImageNet-100 experiments require a dataset root with `train/` and `val/` folders containing the same 100 class directories. Supply the subset you intend to evaluate with `--data-root`. For Class IL you can also set `data_root` in the JSON config. CIFAR-100 uses the existing automatic download.

If there are any questions, please feel free to open an issue or contact the author, **Aojun Lu** ([aojunlu@stu.scu.edu.cn](mailto:aojunlu@stu.scu.edu.cn)).

## How to Use

Run commands from the corresponding scenario directory.

**Task IL**

```sh
cd task_il
python test.py --architecture alexac-a --increment 5
python test.py --architecture alexnet --dataset imagenet100 --data-root /path/to/imagenet100 --increment 10
```

**Class IL**

```sh
cd class_il
python test.py --config exps/replay.json --architecture resac-b --increment 5
python test.py --config exps/foster.json --architecture resnet18 --dataset imagenet100 --data-root /path/to/imagenet100 --increment 10
```

The paper experiment configurations run five seeds and report LA and AIA as mean ± standard deviation. The default seed list is 1993–1997. Use `--seeds 1993` for a single run.

**Architecture search**

Run `python evolve.py` from either scenario directory. The search evaluates an initial population of 10, then runs 20 generations using mean AIA as fitness. Start a fresh run with `is_running = 0` in `global.ini`. Resume only from population files generated with the same search settings. Use a clean `populations/` directory for a new search protocol.

## Citation
If you find this repo useful, please consider citing our paper.
```bibtex
@inproceedings{ArchCraft_Lu_IJCAI2024,
  title = {Revisiting Neural Networks for Continual Learning: An Architectural Perspective},
  author = {Lu, Aojun and Feng, Tao and Yuan, Hangjie and Song, Xiaotian and Sun, Yanan},
  booktitle = {IJCAI},
  pages = {4651--4659},
  year = {2024}
}
```

## Acknowledgement
Part of this work's implementation refers to several prior works including [CNN-GA](https://github.com/yn-sun/cnn-ga), [PyCIL](https://github.com/G-U-N/PyCIL), and [HAT](https://github.com/joansj/hat).
