# Fistula-Segmentation
Segmenting fistulae from abdominal CT scans

** Written during/after our meeting on 8/6 **



To train:
- edit /config/my_config.py file
- activate conda environment
- move to root directory of the project (`cd /my/path/to/Fistula-Segmentation`)
- run `python scripts/train.py config/my_config`


To deploy:
- TODO: create deployment script that takes DICOM(?)
- create script that instantiates network architecture, loads weights, takes in image argument, and passes that to model
