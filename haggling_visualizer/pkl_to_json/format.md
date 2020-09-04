# json format
```json
{
    "buyer_id": u8,
    "seller_ids": [u8; 2],
    "left_seller_id": u8,
    "right_seller_id": u8,
    "winner_id": u8,
    "start_frame": u32,
    "subjects": [
        {
            "human_id": u8,
            "start_frame": u32,
            "b_valid": bool,
            "frames": [
                {
                    "joints19": [f32; 57],
                    "body_normal": [f32; 3],
                    "face_normal": [f32; 3],
                    "scores": [f32; 19],
                }
            ],
        }
    ]
}
```
# index to body part map
| bone | joint1 - joint2 |
| --- | --- |
| [0, 1]    | Neck - Nose           |
| [0, 3]    | Neck - lShoulder      |
| [3, 4]    | lShoulder - lElbow    |
| [4, 5]    | lElbow - lWrist       |
| [0, 2]    | Neck - BodyCenter     |
| [2, 6]    | BodyCenter - lHip     |
| [6, 7]    | lHip - lKnee          |
| [7, 8]    | lKnee - lAnkle        |
| [2, 12]   | BodyCenter - rHip     |
| [12, 13]  | rHip - rKnee          |
| [13, 14]  | rKnee - rAnkle        |
| [0, 9]    | Neck - rShoulder      |
| [9, 10]   | rShoulder - rElbow    |
| [10, 11]  | rElbow - rWrist       |
