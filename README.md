# CleanerS: Semantic Scene Completion with Cleaner Self

[Fengyun Wang](), [Dong Zhang](), [Hanwang Zhang](), [Jinhui Tang](https://scholar.google.com/citations?user=ByBLlEwAAAAJ&hl=zh-CN), [Qianru Sun]()

[IMAG Lab](https://imag-njust.net/), Nanjing University of Science and Technology

---

[![GitHub Stars](https://img.shields.io/github/stars/fereenwong/CleanerS?style=social)](https://github.com/fereenwong/CleanerS)  ![visitors](https://visitor-badge.glitch.me/badge?page_id=fereenwong/CleanerS)  [![arXiv](https://img.shields.io/badge/arXiv-Paper-.svg)]()  ![update](https://badges.strrl.dev/updated/fereenwong/CleanerS)

---

<p align="center">
  <img width="800" src="./figs/framework.png">
</p>

Overall architecture of our proposed CleanerS, consisting of two networks: a teacher network, and a student network. The two networks share same architectures but have different weights. The distillation pipelines include a feature-based cleaner surface distillation (*i.e.*, *KD-T*), and logit-based cleaner semantic distillations (*i.e.*, *KD-SC* and *KD-SA*). The dimensions of the inputs and outputs in the student network are omitted as they are the same as in the teacher network.

### Requirements

(a lower/higher vision may also workable)

> - Pytorch 1.10.1
> - cudatoolkit 11.1
> - mmcv 1.5.0
> - mmsegmentation 0.27.0

The suggested installation steps are:

```angular2html
conda create -n CleanerS python=3.7 -y
conda activate CleanerS
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10/index.html
pip install mmsegmentation==0.27.0
conda install scikit-learn
pip install pyyaml timm tqdm EasyConfig multimethod easydict termcolor shortuuid imageio
```

### Data Preparation

We follow the project of [3D-Sketch](https://github.com/charlesCXK/TorchSSC) for dataset preparing. After preparing, `your_SSC_Dataset` folder should look like this:

````
-- your_SSC_Dataset
   |-- NYU
   |-- TSDF
   |-- Mapping
   |   |-- trainset
   |   |-- |-- RGB
   |   |-- |-- depth
   |   |-- |-- GT
   |   |-- testset
   |   |-- |-- RGB
   |   |-- |-- depth
   |   |-- |-- GT
   |-- NYUCAD
   |-- TSDF
   |   |-- trainset
   |   |-- |-- depth
   |   |-- testset
   |   |-- |-- depth
````

### Training

- **Segformer-B2**

> 1. Download the pretrained Segformer-B2, [mit_b2.pth](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia);
> 2. Run `run.sh` for training the CleanerS (train both teacher and student models);
> 3. (optional) Download the [teacher model]() and put it into `./teacher/Teacher_ckpt.pth`, and then train only the student model with distillation

- **ResNet50**

> 1. Download the pretrained [ResNet50](https://drive.google.com/drive/folders/121yZXBZ8wV77WRXRur86YBA4ifJEhsJQ).
> 2.

### Testing with Our Pretrained model

> 1. Download our [pretrained model]() and then put it in the `./checkpoint` folder.
> 2. Run ``test_NYU.py``. The visualized results will be in the `./visual/test` folder. (``xx.ply`` files)

### Results


| Segformer-B2      |                    Model Zoo                    |                 Visual Results                 |
| :------------------ | :-----------------------------------------------: | :-----------------------------------------------: |
| Teacher Model     | [Google Drive]() / [Baidu Netdisk]() with code: | [Google Drive]() / [Baidu Netdisk]() with code: |
| **Student Model** | [Google Drive]() / [Baidu Netdisk]() with code: | [Google Drive]() / [Baidu Netdisk]() with code: |

## TODO
- [ ] upload pretrained weight and experimental results
- [ ]  replace the segformer as ResNet50
