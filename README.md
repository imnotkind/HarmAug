# HarmAug: Effective Data Augmentation for Knowledge Distillation of Safety Guard Models

[[Model link]](https://huggingface.co/AnonHB/HarmAug_Guard_Model_deberta_v3_large_finetuned)  
[[Dataset link]](https://huggingface.co/datasets/AnonHB/HarmAug_generated_dataset)


![concept_figure](https://github.com/user-attachments/assets/3e61f7c6-e0c2-4107-bb4e-9b4d2c7ba961)

![overall_comparison_broken](https://github.com/user-attachments/assets/03cc0fa5-e9dc-4d78-a5b8-a2c122672fea)


## Reproduction Steps

First, we recommend to create a conda environment with python 3.10.
```
conda create -n harmaug python=3.10
conda activate harmaug
```


After that, install the requirements.
```
pip install -r requirements.txt
```


Then, download necessary files from [Google Drive](https://drive.google.com/drive/folders/1oLUMPauXYtEBP7rvbULXL4hHp9Ck_yqg?usp=drive_link) and put them into their appropriate folders.
```
mv kd_dataset@harmaug.json ./data
```


Finally, you can start the knowledge distillation process.
```
bash script/kd.sh
```

## Reference
To cite our paper, please use this BibTex
```bibtex
@article{lee2024harmaug,
  title={{HarmAug}: Effective Data Augmentation for Knowledge Distillation of Safety Guard Models},
  author={Lee, Seanie and Seong, Haebin and Lee, Dong Bok and Kang, Minki and Chen, Xiaoyin and Wagner, Dominik and Bengio, Yoshua and Lee, Juho and Hwang, Sung Ju},
  journal={arXiv preprint arXiv:2410.01524},
  year={2024}
}
```

