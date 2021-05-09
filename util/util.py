
def cal_psnr(label, outputs, max_val=255.0):
    
    label = label.cpu().detach()
    outputs = outputs.cpu().detach()
    img_diff = outputs - label

    return 10. * ((max_val ** 2) / ((img_diff) ** 2).mean()).log10()
    # rmse = math.sqrt(np.mean((img_diff) ** 2))

    # if rmse == 0:
    #     return 100
    # else:
    #     PSNR = 20 * math.log10(max_val / rmse)
    #     return PSNR

def convert_rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.


def denormalize(img):
    img = img.mul(255.0).clamp(0.0, 255.0)
    return img
