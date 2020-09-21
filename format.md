#JSON format
```json
{
    "winner_id": u8,
    "subjects":
        {
            "buyer": {
                "human_id": u8,
                "frames":
                    {
                        "joints21": [f32; 73],
                        "body_normal": [f32; 2],
                        "face_normal": [f32; 2],
                    },
                "initTrans": f32,
                "initRot": f32
            },
            "leftSeller": {
                "human_id": u8,
                "frames":
                    {
                        "joints21": [f32; 73],
                        "body_normal": [f32; 2],
                        "face_normal": [f32; 2],
                    },
                "initTrans": f32,
                "initRot": f32
            },
            "rightSeller": {
                "human_id": u8,
                "frames":
                    {
                        "joints21": [f32; 73],
                        "body_normal": [f32; 2],
                        "face_normal": [f32; 2],
                    },
                "initTrans": f32,
                "initRot": f32
            }
            "prediction<insert_name>": {
                "human_id": u8,
                "frames":
                    {
                        "joints21": [f32; 73],
                        "body_normal": [f32; 2],
                        "face_normal": [f32; 2],
                    },
                "initTrans": f32,
                "initRot": f32
            }
        }
}
```
# index to body part map
| bone | joint1 - joint2 |
| --- | --- |
| [1, 5]    | lHip - rHip           |
| [0, 9]    | BodyCenter - Torso    |
| [9, 10]   | Torso - Chest         |
| [10, 11]  | Chest - Neck          |
| [11, 12]  | Neck - Nose           |
| [5, 6]    | rHip - rKnee          |
| [1, 2]    | lHip - lKnee          |
| [6, 7]    | rKnee - rAnkle        |
| [2, 3]    | lKnee - lAnkle        |
| [13, 17]  | lShoulder - rShoulder |
| [17, 18]  | rShoulder - rElbow    |
| [13, 14]  | lShoulder - lElbow    |
| [14, 15]  | lElbow - lWrist       |
| [18, 19]  | rElbow - rWrist       |
