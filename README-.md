# CleanerS: Semantic Scene Completion with Cleaner Self

[Fengyun Wang](https://fereenwong.github.io/), [Dong Zhang](https://dongzhang89.github.io/), [Hanwang Zhang](https://personal.ntu.edu.sg/hanwangzhang/), [Jinhui Tang](https://scholar.google.com/citations?user=ByBLlEwAAAAJ&hl=zh-CN), [Qianru Sun](https://qianrusun.com/)

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
> 2. (optional) Download the [teacher model]() and put it into `./teacher/Teacher_ckpt.pth`;
> 3. Run `run.sh` for training the CleanerS (if you skip the step 2, it will train both teacher and student models).

- **ResNet50**

> 1. Download the pretrained [ResNet50](https://drive.google.com/drive/folders/121yZXBZ8wV77WRXRur86YBA4ifJEhsJQ).
> 2.

### Testing with Our Pretrained model

> 1. Download our [pretrained model]() and then put it in the `./checkpoint` folder.
> 2. Run ``python test_NYU.py --pretrained_path ./checkpoint/CleanerS_ckpt.pth``. The visualized results will be in the `./visual_pred/CleanerS` folder.
> 3. (optional) Run ``python test_NYU.py --pretrained_path ./checkpoint/Teacher_ckpt.pth`` to get the results of the teacher model.

### Results


| Segformer-B2      |                    Model Zoo                    |                 Visual Results                 |
| :------------------ | :-----------------------------------------------: | :-----------------------------------------------: |
| Teacher Model     | [Google Drive](https://drive.google.com/file/d/1e8GZRFLMUM9WLoDm3GITJ6YV8solWMfk/view?usp=sharing) / [Baidu Netdisk](https://pan.baidu.com/s/1bc6ODl6VIjRBwgQ7wwypnA?pwd=3gew) with code:3gew | [Google Drive](https://drive.google.com/file/d/1jFCzMBj4l8itpDWzSgXaI8c4kYZlsrLX/view?usp=sharing) / [Baidu Netdisk](https://pan.baidu.com/s/1snrfT0BCX4JiW2hC6pYJnw?pwd=p9nl) with code:p9nl |
| **Student Model** | [Google Drive](https://drive.google.com/file/d/1LyUAPq4WaB-PxyrPZ0L33_a3aKgMK5aW/view?usp=sharing) / [Baidu Netdisk](https://pan.baidu.com/s/1puxavCn3nUr-eguJiqBdDw?pwd=6eja) with code:6eja | [Google Drive](https://drive.google.com/file/d/15jlkRQRp142zmoG7KREB5dBbLDufTqd8/view?usp=sharing) / [Baidu Netdisk](https://pan.baidu.com/s/1Sn0Iq3tEHxcFOG78Vg_6nQ?pwd=lktg) with code:lktg |

- **Comparison with SOTA**
<p align="center">
  <img width="800" src="./figs/Comparison-tab.png">
</p>

### Citation
If this work is helpful for your research, please consider citing the following BibTeX entry.
```
citation
```

## TODO
- [ ] BibTeX for citation
- [ ] arixv link of the paper
- [ ] switchable 2DNet for both Segformer-B2 and ResNet50

### Acknowledgement
This code is based on [3D-Sketch](https://github.com/charlesCXK/TorchSSC). Thanks for the awesome work.
