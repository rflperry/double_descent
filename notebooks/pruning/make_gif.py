import glob
import re
from PIL import Image

# filepaths

direc = './results/spiral_training_epochs/'
name = "./spiral_layers=2_training_hyperplanes"#
fp_in = f"{direc}{name}*.png"
fp_out = f"./reports/{name}.gif"

filepaths = glob.glob(fp_in)
# print(fp_in[:fp_in.index("*")] + r"([0-9]+)*.png")
sorted_filepaths = sorted(
    filepaths, key = lambda s: int(re.match(fp_in[:fp_in.index("*")] + r".*?([0-9]+)\.png", s).group(1))
)[:100]#[::-1]

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted_filepaths]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=120, loop=0)
