from idyom import idyom, data
from App import Train, SurpriseOverFolder, cross_validation
import mido
import collections
import numpy as np

train_path = "trainingFolder/"
eval_path = "testFolder/"

quantization = 24
maxOrder = 20
viewPoints = ["pitch", "length"]

midiFile = "stimuli/eLife/audio1.mid"

# 1. Train the model
Train(
    folder=train_path,
    quantization=quantization,
    maxOrder=maxOrder,
    viewPoints=viewPoints,
)

# 2. Load model
train_name = train_path.rstrip("/").split("/")[-1]
model_path = f"models/{train_name}_quantization_{quantization}_maxOrder_{maxOrder}_viewpoints_{viewPoints[0]}_{viewPoints[1]}.model"

model = idyom.idyom()
model.load(model_path)

# 3. Surprise for a single file
IC = model.getSurprisefromFile(midiFile)

# 4. Surprise over folder
SurpriseOverFolder(
    folderTrain=train_path,
    folder=eval_path,
    quantization=24,
    maxOrder=20,
)

# 5. Cross-validation - still not working... probably referencing the wrong folder or need to re-load results that were saved from the SurpriseOverFolder function
ICs, Entropies, val_files = cross_validation(
    folder=eval_path,  
    k_fold=10,
    maxOrder=20,
    quantization=24
)

