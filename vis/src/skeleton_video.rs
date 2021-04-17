use bevy::prelude::*;
use serde::{de, Deserialize, Deserializer};
use serde_json::Value;
use std::fmt;

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct SkeletonVideo {
    id: u8,
    frames: Vec<SkeletonFrame>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct SkeletonFrame {
    #[serde(deserialize_with = "de_joints21")]
    joints21: [Vec3; 21],
}

fn de_joints21<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[Vec3; 21], D::Error> {
    Ok(match Value::deserialize(deserializer)? {
        Value::Array(array) => {
            if array.len() == 63 {
                let mut ans: [Vec3; 21] = [Vec3::ZERO; 21];
                for idx in 0..ans.len() {
                    ans[idx] = Vec3::new(
                        array[3 * idx]
                            .as_f64()
                            .ok_or(de::Error::custom("Invalid number"))?
                            as f32,
                        array[3 * idx + 1]
                            .as_f64()
                            .ok_or(de::Error::custom("Invalid number"))?
                            as f32,
                        array[3 * idx + 2]
                            .as_f64()
                            .ok_or(de::Error::custom("Invalid number"))?
                            as f32,
                    );
                }
                ans
            } else {
                return Err(de::Error::custom(
                    "Invalid frame. Did not find exactly 63 numbers per frame",
                ));
            }
        }
        _ => return Err(de::Error::custom("Wrong type for frame")),
    })
}

impl SkeletonVideo {
    pub const THICKNESS: f32 = 0.2;
    const SCALE: f32 = 0.1;
    const BONES: [(usize, usize); 17] = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (10, 13),
        (13, 14),
        (14, 19),
        (10, 17),
        (17, 18),
        (18, 15),
    ];

    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    fn get_transform(&self, idx1: usize, idx2: usize, frame_number: usize) -> (f32, Vec3, Quat) {
        let p1 = self.frames[frame_number].joints21[idx1] * SkeletonVideo::SCALE;
        let p2 = self.frames[frame_number].joints21[idx2] * SkeletonVideo::SCALE;
        (
            (p2 - p1).length(),
            (p1 + p2) / 2.0,
            Quat::from_rotation_arc(Vec3::X, (p2 - p1).normalize()),
        )
    }

    pub fn bones(&self, frame_number: usize) -> [(f32, Vec3, Quat); 17] {
        let mut bones = [(0.0, Vec3::ZERO, Quat::IDENTITY); 17];
        for (idx, (p1, p2)) in SkeletonVideo::BONES.iter().enumerate() {
            bones[idx] = self.get_transform(*p1, *p2, frame_number);
        }
        bones
    }

    pub fn bone(&self, bone_idx: usize, frame_number: usize) -> (f32, Vec3, Quat) {
        self.get_transform(
            SkeletonVideo::BONES[bone_idx].0,
            SkeletonVideo::BONES[bone_idx].1,
            frame_number,
        )
    }
}

impl fmt::Display for SkeletonVideo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "
    id\t\t={}
    #frames\t\t={}",
            self.id,
            self.frames.len(),
        )
    }
}
