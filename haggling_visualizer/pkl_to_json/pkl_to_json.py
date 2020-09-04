import pickle
import numpy as np
import json
import sys

if len(sys.argv) != 2:
    print("Usage: python3 {} <pkl-file>".format(sys.argv[0]))
    exit(-1)

pkl_file_path = sys.argv[1]

print("Converting {}".format(pkl_file_path))

with open(pkl_file_path, "rb") as f:
    old_format = pickle.load(f, encoding="bytes")

new_format = {
    "buyer_id": old_format[b"buyerId"],
    "seller_ids": old_format[b"sellerIds"],
    "left_seller_id": old_format[b"leftSellerId"],
    "right_seller_id": old_format[b"rightSellerId"],
    "winner_id": old_format[b"winnerId"],
    "start_frame": old_format[b"startFrame"],
}

def convert_subject_to_new_format(old_format_subject):
    new_format_subject = {
        "human_id": old_format_subject[b"humanId"],
        "start_frame": old_format_subject[b"startFrame"],
        "b_valid": old_format_subject[b"bValid"],
    }
    # Convert from (per_frame_len x num_frames) to (num_frames x per_frame_len) for each of the following
    joints19 = np.transpose(old_format_subject[b"joints19"]).tolist()
    body_normal = np.transpose(old_format_subject[b"bodyNormal"]).tolist()
    face_normal = np.transpose(old_format_subject[b"faceNormal"]).tolist()
    scores = np.transpose(old_format_subject[b"scores"]).tolist()
    # Num. of frames for each of these should be equal
    assert(len(joints19) == len(body_normal) == len(face_normal) == len(scores))
    frames = []
    for i in range(len(joints19)):
        frames.append({
                "joints19": joints19[i],
                "body_normal": body_normal[i],
                "face_normal": face_normal[i],
                "scores": scores[i],
            })
    new_format_subject["frames"] = frames
    return new_format_subject

new_format["subjects"] = list(map(convert_subject_to_new_format, old_format[b"subjects"]))

json_file_path = "{}.json".format(pkl_file_path)

# Write to json
with open(json_file_path, "w") as f:
    json.dump(new_format, f)

print("Exploring {}".format(json_file_path))

# Read from written file
with open("{}".format(json_file_path),"rb") as f:
    parsed_json = json.load(f)

print(type(parsed_json))
print(parsed_json.keys())
print(len(parsed_json.keys()))

print("1. buyer_id={}".format(parsed_json["buyer_id"]))
print("2. seller_ids={}".format(parsed_json["seller_ids"]))
print("3. left_seller_id={}".format(parsed_json["left_seller_id"]))
print("4. right_seller_id={}".format(parsed_json["right_seller_id"]))
print("5. winner_id={}".format(parsed_json["winner_id"]))
print("6. start_frame={}".format(parsed_json["start_frame"]))

print("7. subjects type={}".format(type(parsed_json["subjects"])))
print("   subjects len={}".format(len(parsed_json["subjects"])))

print("   {}".format(parsed_json["subjects"][0].keys()))
print("   {}".format(parsed_json["subjects"][1].keys()))
print("   {}".format(parsed_json["subjects"][2].keys()))

print("   for 0th subject")
print("      human_id={}".format(np.asarray(parsed_json["subjects"][0]["human_id"])))
print("      start_frame={}".format(np.asarray(parsed_json["subjects"][0]["start_frame"])))
print("      b_valid={}".format(np.asarray(parsed_json["subjects"][0]["b_valid"])))
print("      frames len={}".format(len(parsed_json["subjects"][0]["frames"])))
print("      0th frame keys={}".format(parsed_json["subjects"][0]["frames"][0].keys()))
