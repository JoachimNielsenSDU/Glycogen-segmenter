# Glycogen-segmenter
A model for automatically predicting glycogen within distinct subcellular regions of skeletal muscle fibres using transmission electron microscopy (TEM) images.
The raw images, annotated masks, and model weights are available here: https://zenodo.org/uploads/18390286 (DOI: 10.5281/zenodo.18390286)

How to make the GUI work on Windows:
1) Download the model weights (region and glycogen) from https://zenodo.org/uploads/18390286
2) Create a folder called "Glycogen-segmenter"
3) Inside this folder, add the weights to two subfolders called "weights_region" and "weights_glycogen", respectively
4) Add unet.py, requirements.txt, and gui_glycogen_segmenter_v996.py to the "Glycogen-segmenter"-folder
5) Open your terminal (e.g. Anaconda Prompt)
6) Run the following commmands:
      - cd "Path to your folder here"
      - conda create --name glyco python=3.11
      - conda activate glyco
      - pip install -r requirements.txt
      - python gui_glycogen_segmenter_v996.py
7) The GUI should open in your web browser
Note: The model was trained only on myofibrillar images. However, it can also predict subsarcolemmal glycogen (classified as intermyofibrillar) when applied to           subsarcolemmal images. The model can additionally predict mitochondria and Z-disc width, which may be used as fibre type indicators.

How the model was trained:
Two models were trained: one for predicting subcellular regions and one for predicting glycogen. The combination of these two models allows prediction of glycogen within specific subcellular regions. When train.py (region model) and train_glycogen.py (glycogen model) are run, they call: unet.py, datagenerator.py, loss.py, and nrrdreader.py. Training uses folders containing raw images (TIF files) and annotations (.seg.nrrd files) for regions (region.zip) and glycogen (glycogen.zip), respectively. These data are available at: https://zenodo.org/uploads/18390286
