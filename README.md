## <div align="center">Arcface</div>

This repo focus implement the paper with Pytorch [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698.pdf).

<a align="center" href="https://arxiv.org/pdf/1801.07698.pdf" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/phuocdangday/arcface-pytorch/master/examples/arcface.jpg"></a>

## <div align="center">Documentation</div>

<details open>
<summary>Install</summary>

```bash
git clone https://github.com/phuocdangday/arcface-pytorch  # clone
cd arcface-pytorch
pip install -r requirements.txt  # install requirements
```
</details>

<details open>
<summary>Training</summary>

Data structure:

```
arcface-pytorch
├── data/
│   ├── train/
│   │   ├── class_a/
│   │   │   ├── a_image_1.jpg
│   │   │   ├── a_image_2.jpg
│   │   │   └── a_image_3.jpg
│   │   ├── class_b/
│   │   │   ├── b_image_1.jpg
│   │   │   ├── b_image_2.jpg
│   │   │   └── b_image_3.jpg
│   │   └── class_c/
│   │       ├── c_image_1.jpg
│   │       ├── c_image_2.jpg
│   │       └── c_image_3.jpg
│   └── valid/
│       ├── class_a/
│       │   ├── a_image_1.jpg
│       │   ├── a_image_2.jpg
│       │   └── a_image_3.jpg
│       ├── class_b/
│       │   ├── b_image_1.jpg
│       │   ├── b_image_2.jpg
│       │   └── b_image_3.jpg
│       └── class_c/
│           ├── c_image_1.jpg
│           ├── c_image_2.jpg
│           └── c_image_3.jpg
└── train.py
```

```bash
python train.py --epochs 300 --learning-rate 1e3 --data data --batch-size 128 --image-size 224 --embedding-size 512 --margin-loss 0.3 --scale-loss 30 --num-workers 1 --device cuda:0
```

</details>
