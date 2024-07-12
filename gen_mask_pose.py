import argparse
import json
import os
import numpy as np
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from PIL import Image
import apply_net as apply_net
from utils_mask import get_mask_location
def parse_args():
    parser = argparse.ArgumentParser(description="proc Training data script for IDM-VTON.")
  
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--phase", type=str, default="train",help="phase must be \'train\' or \'test\'!")
    parser.add_argument("--model_type",type=str,help="model_type must be \'hd\' or \'dc\'!", default="hd")
    parser.add_argument("--force",type=bool,default=False)
    args = parser.parse_args()
    return args

# https://en.wikipedia.org/wiki/YUV#SDTV_with_BT.601
_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
_M_YUV2RGB = [[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]]

# https://www.exiv2.org/tags.html
_EXIF_ORIENT = 274  # exif 'Orientation' tag
def convert_PIL_to_numpy(image, format):
    """
    Convert PIL image to numpy array of target format.

    Args:
        image (PIL.Image): a PIL image
        format (str): the format of output image

    Returns:
        (np.ndarray): also see `read_image`
    """
    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format in ["BGR", "YUV-BT.601"]:
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)

    # handle formats not supported by PIL
    elif format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    elif format == "YUV-BT.601":
        image = image / 255.0
        image = np.dot(image, np.array(_M_RGB2YUV).T)

    return image
def _apply_exif_orientation(image):
    """
    Applies the exif orientation correctly.

    This code exists per the bug:
      https://github.com/python-pillow/Pillow/issues/3973
    with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
    various methods, especially `tobytes`

    Function based on:
      https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
      https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527

    Args:
        image (PIL.Image): a PIL image

    Returns:
        (PIL.Image): the PIL image with exif orientation applied, if applicable
    """
    if not hasattr(image, "getexif"):
        return image

    try:
        exif = image.getexif()
    except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
        exif = None

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        return image.transpose(method)
    return image

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask


def main():
    args = parse_args()
    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)
    with open(os.path.join(args.data_dir,f"garment_desc_{args.phase}.txt"), "r") as file1:
        data = json.load(file1)
        for item in data:
            print(item)
            has_mask=os.path.exists(os.path.join(args.data_dir,"garments",item["type"],"mask_"+item["img_fn"]))
            has_pose=os.path.exists(os.path.join(args.data_dir,"models","pose_"+item["img_fn"]))
            human_img=None
            if not (has_mask and has_pose) or args.force:
                human_img=Image.open(os.path.join(args.data_dir,"models",item["img_fn"]))
                print(os.path.join(args.data_dir,"models",item["img_fn"]))

            if not has_mask or args.force:
                try:
                    keypoints = openpose_model(human_img.resize((384,512)))
                    model_parse, _ = parsing_model(human_img.resize((384,512)))
                    mask, _ = get_mask_location(args.model_type, item["type"], model_parse, keypoints)
                    mask = mask.resize((768,1024))
                    mask.save(os.path.join(args.data_dir,"garments",item["type"],"mask_"+item["img_fn"]))
                    print(os.path.join(args.data_dir,"garments",item["type"],"mask_"+item["img_fn"]))
                except:
                    print("生成模板失败!",item)
            

            if not has_pose or args.force:
                try:
                    human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
                    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
                    fargs = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
                    # verbosity = getattr(args, "verbosity", None)
                    pose_img = fargs.func(fargs,human_img_arg)    
                    pose_img = pose_img[:,:,::-1]    
                    pose_img = Image.fromarray(pose_img).resize((768,1024))
                    pose_img.save(os.path.join(args.data_dir,"models","pose_"+item["img_fn"]))
                    print(os.path.join(args.data_dir,"models","pose_"+item["img_fn"]))
                except:
                    print("生成姿势图片失败！",item)


            


if __name__ == "__main__":
    main()