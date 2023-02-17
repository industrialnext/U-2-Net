import argparse
import os

import json

source_ref_key = "source-ref"
suffix = "ref"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename label files")
    parser.add_argument("-m", required=True, action="store", help="annotations.manifest")
    parser.add_argument("-l", required=True, action="store", help="directory to labels")
    args = parser.parse_args()

    if not os.path.isfile(args.m) or not os.path.isdir(args.l):
        print("Paths don't exist!")
        exit()

    with open(args.m) as f:
        manifest = f.read()

    m = manifest.split()
    print(f"File count in manifest: {len(m)}")

    # Iterate through all entries
    for item in m:
        # Convert to dict
        d = json.loads(item)
        keys = list(d.keys())

        if keys[0] == source_ref_key and keys[1].endswith(suffix): 
            output_key = keys[1]
        else:
            print(f"Cant find keys in {d}, abort!")
            break

        data_file_name = d[source_ref_key].split("/")[-1]
        data_file_no_ext = os.path.splitext(data_file_name)[0]
        label_file_name = d[output_key].split("/")[-1]
        label_file_ext = os.path.splitext(label_file_name)[-1]
        print(f"Rename labeled output file: {label_file_name} -> {data_file_no_ext+label_file_ext}")

        try:
            os.rename(args.l+"/"+label_file_name, args.l+"/"+data_file_no_ext+label_file_ext)
        except Exception as e:
            print(e)

