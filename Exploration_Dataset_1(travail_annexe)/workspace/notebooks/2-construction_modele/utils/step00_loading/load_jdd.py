import xml.etree.ElementTree as ET
from pathlib import Path
import os
import base64
from io import BytesIO
from ipywidgets import IntProgress
import zipfile
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# FORMATS IMAGE SUPPORTES
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    
# Load Image
def load_image_to_base64(z: zipfile.ZipFile, image_path):
    with z.open(image_path) as f:
        data = f.read()
        return  base64.b64encode(data)

# Load depuis ZIP File
def load_images_to_dataframe(z: zipfile.ZipFile) -> pd.DataFrame:
    image_paths = []
    image_files = []
    image_types = []
    images_base64 = []
    widths = []
    heights = []

    progress = IntProgress(
        value=0,
        min=0,
        max=len(z.namelist()),
        description='Load images',
        bar_style='info',
        style={'description_width': '200px'}
    )
    display(progress)
    for f in z.namelist():
        progress.value += 1
        progress.description= os.path.basename(f) + ' is loading...'
        if f.lower().endswith(IMAGE_EXTENSIONS):
            img = load_image_to_base64(z, f)
            image_paths.append(f)
            image_files.append(os.path.basename(f))
            image_types.append(os.path.splitext(f)[1].lower().replace(".", ""))
            images_base64.append(img)
    
    progress.layout.display = 'none'
        
    
    df = pd.DataFrame({ 
        "image_filename": image_files,
        "image_type": image_types,
        "image_base64": images_base64,
    })
    
    return df

def load_jdd_zippe(path_zipfile):
    z = zipfile.ZipFile(path_zipfile, "r")
    return load_images_to_dataframe(z)