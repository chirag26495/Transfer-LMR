# Tranfer-LMR for Driving Behavior Recognition

```bash
git clone https://github.com/piergiaj/pytorch-i3d.git
git clone https://github.com/chirag26495/Transfer-LMR.git
cp -r Transfer-LMR/* pytorch-i3d/
cd pytorch-i3d/
```

## Environment setup (for Python 3.10.5):
```bash
conda env create -f environment.yml
pip3 install -r requirements.txt
```

## Setup [METEOR](https://github.com/GAMMA-UMD/METEOR.git) dataset.
- Create a RawFrames format dataset
- Dataset location: "datasets/meteor/original_data/rawframe_format_fullres/"

## Train & Validate I3D-based CE baseline:
```bash
python3 train_i3d_rgb_meteor.py
```

## Train & Validate I3D-based Transfer-LMR approach:
```bash
python3 train_i3d_TFAavg_wLMR_rgb_meteor.py
```


## Repo References

**[pytorch-i3d](https://github.com/piergiaj/pytorch-i3d)**

**[MMAction2](https://github.com/open-mmlab/mmaction2)**


