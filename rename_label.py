import argparse
import os
import cv2
import json

source_ref_key = "source-ref"
suffix = "ref"
image_exts = (".png", ".jpg")

def process_label(manifest, label_dir, if_binarize):
    try:
        with open(manifest) as f:
            manifest = f.read()
    except Exception as e:
        print(e)
        return False

    m = manifest.split()
    print(f"File count in manifest: {len(m)}")
    print(f"Binarize label: {if_binarize}")

    count = 0
    # Iterate through all entries
    for item in m:
        # Convert to dict
        d = json.loads(item)
        keys = list(d.keys())

        if keys[0] == source_ref_key and keys[1].endswith(suffix): 
            output_key = keys[1]
        else:
            print(f"Cant find keys in {d}, abort!")
            return False

        data_file_name = d[source_ref_key].split(os.sep)[-1]
        data_file_no_ext = os.path.splitext(data_file_name)[0]
        label_file_name = d[output_key].split(os.sep)[-1]
        label_file_ext = os.path.splitext(label_file_name)[-1]
        #print(f"Rename labeled output file: {label_file_name} -> {data_file_no_ext+label_file_ext}")

        label_file_path = os.path.join(label_dir, data_file_no_ext+label_file_ext)
        try:
            # Rename label
            os.rename(os.path.join(label_dir, label_file_name), label_file_path)

            if if_binarize:
                # Binarize labels
                binary_label = binarize_image(label_file_path)
                cv2.imwrite(label_file_path, binary_label)
                # FOR DEBUGGING #
                #cv2.imshow("binary", binary_label)
                #cv2.waitKey(0)
                #break
            count+=1
        except Exception as e:
            print(e)

    print(f"Processed {count} labels")
    return True

def binarize_all_images(label_dir):
    print("Binarize all")
    count = 0

    for image_file in os.listdir(label_dir):
        for ext in image_exts:
            if (image_file.endswith(ext)):
                image_path = os.path.join(label_dir, image_file)
                cv2.imwrite(image_path, binarize_image(os.path.join(label_dir, image_file)))
                count+=1

    print(f"Image count: {count}")

# This turns background black and label foreground white
def binarize_image(img_path):
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    ret, binary_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)

    return binary_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename label files")
    parser.add_argument("-m", action="store", help="annotations.manifest. Provide the manifest if you would like rename labels")
    parser.add_argument("-l", required=True, action="store", help="label directory")
    parser.add_argument("-b", action="store_true", help="binarize labels")
    args = parser.parse_args()

    if not os.path.isdir(args.l):
        print("Label directory does not exist!")
    elif args.m:
        if not os.path.isfile(args.m):
            print("Manifest file does not exist!")
        elif process_label(args.m, args.l, args.b):
            print("Success")
        else:
            print("Process failed")
    elif args.b:
        binarize_all_images(args.l)
    else:
        print("No operation done")
