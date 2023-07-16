Firstly, this code is improved from YOLOX. Thanks very much to them for providing the YOLOX's code.
```latex
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
1. The dataset we used is SSDD(SAR Ship Detection Dataset)[1], it's in ./datasets/VOCdevkit.
2. The trained network weights are in ./training weights, 
where ID=1 represents the first ablation experiment, etc. 
3. Our model is in ./yolox/models_ID=i, i=1,2,3,4, which represents the five groups of ablation experiments. 
4. If you want to train our model, please run train.py, which is in ./tools. 
5. If you want to evaluate our model, please run eval.py, which is in ./tools. 
6. And if you want to predict SAR images by using our model, please run demo.py, which is also in ./tools. 

Reference
[1]Li, J., Qu, C., & Shao, J. (2017). Ship detection in SAR images based on an improved faster R-CNN. 
2017 SAR in Big Data Era: Models, Methods and Applications (BIGSARDATA). doi:10.1109/bigsardata.2017.8124934