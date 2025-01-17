import os
import glob
import cv2
import shutil
import traceback

def convert(images_dir, output_dir, color_converter):
  image_files = glob.glob(images_dir + "/*.jpg")
  for image_file in image_files:
    basename = os.path.basename(image_file)
    image = cv2.imread(image_file)
    hsvimage = cv2.cvtColor(image, color_converter)

    output_file = os.path.join(output_dir, basename)

    cv2.imwrite(output_file, hsvimage)
    print("--- Saved {}".format(output_file))

if __name__ == "__main__":
  try:
     images_dir = "./mini_test/images"
     output_dir = "./hsv_full_image"
     if os.path.exists(output_dir):
       shutil.rmtree(output_dir)
     os.makedirs(output_dir)
     convert(images_dir, output_dir, cv2.COLOR_BGR2HSV_FULL)

  except:
    traceback.print_exc()
 
