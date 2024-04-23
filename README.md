# Wonder3D
Single Image to 3D using Cross-Domain Diffusion
## [Paper](https://arxiv.org/abs/2310.15008) | [Project page](https://www.xxlong.site/Wonder3D/) | [Hugging Face Demo](https://huggingface.co/spaces/flamehaze1115/Wonder3D-demo)


Wonder3D reconstructs highly-detailed textured meshes from a single-view image in only 2 ∼ 3 minutes. Wonder3D first generates consistent multi-view normal maps with corresponding color images via a cross-domain diffusion model, and then leverages a novel normal fusion method to achieve fast and high-quality reconstruction.

### Preparation for inference

#### Linux System Setup.
```angular2html
conda create -n wonder3d
conda activate wonder3d
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Inference
1. Optional. If you have troubles to connect to huggingface. Make sure you have downloaded the following models.
Download the [checkpoints](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xxlong_connect_hku_hk/Ej7fMT1PwXtKvsELTvDuzuMBebQXEkmf2IwhSjBWtKAJiA) and into the root folder.

If you are in mainland China, you may download via [aliyun](https://www.alipan.com/s/T4rLUNAVq6V).

```bash
Wonder3D
|-- ckpts
    |-- unet
    |-- scheduler
    |-- vae
    ...
```
Then modify the file ./configs/mvdiffusion-joint-ortho-6views.yaml, set `pretrained_model_name_or_path="./ckpts"`

2. Download the [SAM](https://huggingface.co/spaces/abhishek/StableSAM/blob/main/sam_vit_h_4b8939.pth) model. Put it to the ``sam_pt`` folder.
```
Wonder3D
|-- sam_pt
    |-- sam_vit_h_4b8939.pth
```
3. Predict foreground mask as the alpha channel. We use [Clipdrop](https://clipdrop.co/remove-background) to segment the foreground object interactively. 
You may also use `rembg` to remove the backgrounds.
```bash
# !pip install rembg
import rembg
result = rembg.remove(result)
result.show()
```
4. Run Wonder3d to produce multiview-consistent normal maps and color images. Then you can check the results in the folder `./outputs`. (we use `rembg` to remove backgrounds of the results, but the segmentations are not always perfect. May consider using [Clipdrop](https://clipdrop.co/remove-background) to get masks for the generated normal maps and color images, since the quality of masks will significantly influence the reconstructed mesh quality.) 
```bash
accelerate launch --config_file 1gpu.yaml test_mvdiffusion_seq.py \
            --config configs/mvdiffusion-joint-ortho-6views.yaml validation_dataset.root_dir={your_data_path} \
            validation_dataset.filepaths=['your_img_file'] save_dir={your_save_path}
```

see example:

```bash
accelerate launch --config_file 1gpu.yaml test_mvdiffusion_seq.py \
            --config configs/mvdiffusion-joint-ortho-6views.yaml validation_dataset.root_dir=./example_images \
            validation_dataset.filepaths=['owl.png'] save_dir=./outputs
```

5. Mesh Extraction

#### Instant-NSR Mesh Extraction

```bash
cd ./instant-nsr-pl
python launch.py --config configs/neuralangelo-ortho-wmask.yaml --gpu 0 --train dataset.root_dir=../{your_save_path}/cropsize-{crop_size}-cfg{guidance_scale:.1f}/ dataset.scene={scene}
```

see example:

```bash
cd ./instant-nsr-pl
python launch.py --config configs/neuralangelo-ortho-wmask.yaml --gpu 0 --train dataset.root_dir=../outputs/cropsize-192-cfg1.0/ dataset.scene=owl
```