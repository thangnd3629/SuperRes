import torch

from PIL import Image, ImageDraw, ImageFont
from model.myrdn import ResidualDenseNet as RDN
from torchvision.utils import save_image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#hyper param
num_features=64
growth_rate = 64
num_blocks = 16
num_layers = 8
# Data parameters
scaling_factor = 2  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor



rdn = RDN( num_blocks=num_blocks, growth_rate=growth_rate, scale=scaling_factor, num_layers=num_layers, shallow_feature=num_features)
# print(rdn.state_dict().keys())

try:
    rdn.load_state_dict(torch.load(r"checkpoint\rdn_x2.pth", map_location=device))
except Exception as e:
    print(e)

def visualize_sr(img, halve=False):

    # Load image, downsample to obtain low-res version
    hr_img = Image.open(img, mode="r")
    hr_img = hr_img.convert('RGB') 
    
    lr_img = hr_img.resize((int(hr_img.width / scaling_factor), int(hr_img.height / scaling_factor)),
                           Image.BICUBIC)

    # Bicubic Upsampling
    bicubic_img = lr_img.resize((hr_img.width, hr_img.height), Image.BICUBIC)

    
    # Super-resolution (SR) with rdn
    with torch.no_grad():
      rdn.eval()
      sr_img_rdn = rdn(torch.from_numpy(np.asarray(lr_img) / 255.0).permute((2,0,1)).float().unsqueeze(dim = 0).to(device))
      sr_img_rdn = sr_img_rdn.squeeze(0).permute((1,2,0)).cpu().detach().numpy() * 255
      
    
      sr_img_rdn = Image.fromarray(sr_img_rdn.astype('uint8') , 'RGB')
    



    # Create grid
    margin = 40
    grid_img = Image.new('RGB', (2 * hr_img.width + 3 * margin, 2 * hr_img.height + 3 * margin), (255, 255, 255))

    # Font
    draw = ImageDraw.Draw(grid_img)
    try:
        font = ImageFont.truetype("calibril.ttf", size=23)
        # It will also look for this file in your OS's default fonts directory, where you may have the Calibri Light font installed if you have MS Office
        # Otherwise, use any TTF font of your choice
    except OSError:
        print(
            "Defaulting to a terrible font. To use a font of your choice, include the link to its TTF file in the function.")
        font = ImageFont.load_default()

    # Place bicubic-upsampled image
    grid_img.paste(bicubic_img, (margin, margin))
    text_size = font.getsize("Bicubic")
    draw.text(xy=[margin + bicubic_img.width / 2 - text_size[0] / 2, margin - text_size[1] - 5], text="Bicubic",
              font=font,
              fill='black')

    # Place rdn image
    grid_img.paste(sr_img_rdn, (2 * margin + bicubic_img.width, margin))
    text_size = font.getsize("Residual Dense Network")
    draw.text(
        xy=[2 * margin + bicubic_img.width + sr_img_rdn.width / 2 - text_size[0] / 2, margin - text_size[1] - 5],
        text="Residual Dense Network", font=font, fill='black')

    # place lr img
    grid_img.paste(lr_img, (margin + int(bicubic_img.width/2 - lr_img.width/2), 2*margin + int(1.5 * bicubic_img.height - lr_img.height/2)))
    text_size = font.getsize("Low res")
    draw.text(xy=[ margin + int(bicubic_img.width/2 - text_size[0]/2),
                  2 * margin + bicubic_img.height - text_size[1] - 1 + int(lr_img.height/2)] , text="Low res", font=font, fill='black')


    # Place original HR image
    grid_img.paste(hr_img, ( 2*margin + int(bicubic_img.width), 2 * margin + sr_img_rdn.height))
    text_size = font.getsize("Ground Truth HR")
    draw.text(xy=[ 2*margin + bicubic_img.width + sr_img_rdn.width / 2 - text_size[0] / 2,
                  2 * margin + sr_img_rdn.height - text_size[1] - 1], text="Ground Truth HR", font=font, fill='black')

    # Display grid
    grid_img.show()
    grid_img.save('x.png')
    
    return grid_img


visualize_sr(r"asset\192eae919cd5b915b6056e4f270803cd.jpg")