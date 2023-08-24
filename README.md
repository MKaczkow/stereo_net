# stereo_net
Repo for implementation of StereoNet

### TODO
- [x] Get models .ckpt from original repo
- [x] Create gradio UI with multiple IO and option to select best model
- [x] Integrate UI with rest of the code (inference function)
- [ ] Heavy refactor and more DRY
- [ ] Research and fix why KITTIs and dome FlyingThing are not working

First of all, I would like to acknowledge the authors of the paper. I am not the author of the paper, I am was just trying to re-implement it. Link to the paper:  
https://arxiv.org/abs/1807.08865.   

I also relied heavy on this awesome repos:  
https://github.com/andrewlstewart/StereoNet_PyTorch  
https://github.com/zhixuanli/StereoNet  

Contary to mentioned sources I also used both KITTI datasets (KITTI 2012 and KITTI 2015). This was a little chalenging, because of the new policy regarding usage of said datasets, as specified on:   
https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo
```
Important Policy Update: As more and more non-published work and re-implementations 
of existing work is submitted to KITTI, we have established a new policy: from now 
on, only submissions with significant novelty that are leading to a peer-reviewed 
paper in a conference or journal are allowed. Minor modifications of existing 
algorithms or student research projects are not allowed. Such work must be evaluated 
on a split of the training set. To ensure that our policy is adopted, new users must 
detail their status, describe their work and specify the targeted venue during 
registration. Furthermore, we will regularly delete all entries that are 6 months 
old but are still anonymous or do not have a paper associated with them. For 
conferences, 6 month is enough to determine if a paper has been accepted and to add 
the bibliography information. For longer review cycles, you need to resubmit your 
results.
```
This resulted in c.a. 200 pictures from each KITTI dataset, for training, validation and testing. Such little number of pictures resulted in overfitting problems for both KITTI datasets. 

Some of the solutions are probably clunky and not really elegant, but generally gets the job done.

README's for specific folders are in the folders themselves:  
* [data](data)
* [src](src)
* [src.datasets](src/datasets)
* [src.model](src/model)
* [src.utils](src/utils)
