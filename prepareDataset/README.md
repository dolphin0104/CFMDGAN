# Dataset Preparation

## 1) Download 300VW datset training videos from [HERE](https://ibug.doc.ic.ac.uk/resources/300-VW/)

## 2) Extract sharp frames from videos

## 3) Run genBlurDataset.py
+ Specify several paths you want to set in "genBlurDataset.py".
```genBlurDataset.py
line 10 to 13: 
    - origin_dir : 'your/path/to/300VW Dataset'
    - extract_dir : 'your/path/to/300VW Dataset/extract frames'
    - sharp_dir : 'your/path/to/save/sharp ground-truth images'
    - blur_dir : 'your/path/to/save/blur input images'
```

```run
python genBlurDataset.py
```

