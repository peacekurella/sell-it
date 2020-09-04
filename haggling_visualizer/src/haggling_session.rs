use amethyst::core::math::Point3;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Serialize, Deserialize, Debug, Default)]
#[serde(deny_unknown_fields)]
pub struct HagglingSession {
    buyer_id: u8,
    seller_ids: [u8; 2],
    left_seller_id: u8,
    right_seller_id: u8,
    winner_id: u8,
    start_frame: u32,
    subjects: [HagglingSessionSubject; 3],
}

impl HagglingSession {
    pub fn left_right_ids(&self) -> (u8, u8) {
        (self.left_seller_id, self.right_seller_id)
    }

    pub fn subjects(&self) -> &[HagglingSessionSubject] {
        &self.subjects
    }
}

#[derive(Serialize, Deserialize, Debug, Default)]
#[serde(deny_unknown_fields)]
pub struct HagglingSessionSubject {
    human_id: u8,
    start_frame: u32,
    b_valid: bool,
    frames: Vec<Frame>,
}

#[derive(Serialize, Deserialize, Debug, Default)]
#[serde(deny_unknown_fields)]
struct Frame {
    joints19: Vec<f32>,
    body_normal: [f32; 3],
    face_normal: [f32; 3],
    scores: [f32; 19],
}

impl HagglingSessionSubject {
    pub fn human_id(&self) -> u8 {
        self.human_id
    }

    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    pub fn pose_at(&self, frame: usize) -> Option<[(Point3<f32>, Point3<f32>); 14]> {
        if frame > self.frames.len() {
            return None;
        }
        let joints = &self.frames[frame].joints19;
        fn y_inverted_p3(list: &[f32], index: usize) -> Point3<f32> {
            Point3::new(list[index * 3], -list[index * 3 + 1], list[index * 3 + 2])
        }
        Some([
            (y_inverted_p3(&joints, 0), y_inverted_p3(&joints, 1)),
            (y_inverted_p3(&joints, 0), y_inverted_p3(&joints, 3)),
            (y_inverted_p3(&joints, 3), y_inverted_p3(&joints, 4)),
            (y_inverted_p3(&joints, 4), y_inverted_p3(&joints, 5)),
            (y_inverted_p3(&joints, 0), y_inverted_p3(&joints, 2)),
            (y_inverted_p3(&joints, 2), y_inverted_p3(&joints, 6)),
            (y_inverted_p3(&joints, 6), y_inverted_p3(&joints, 7)),
            (y_inverted_p3(&joints, 7), y_inverted_p3(&joints, 8)),
            (y_inverted_p3(&joints, 2), y_inverted_p3(&joints, 12)),
            (y_inverted_p3(&joints, 12), y_inverted_p3(&joints, 13)),
            (y_inverted_p3(&joints, 13), y_inverted_p3(&joints, 14)),
            (y_inverted_p3(&joints, 0), y_inverted_p3(&joints, 9)),
            (y_inverted_p3(&joints, 9), y_inverted_p3(&joints, 10)),
            (y_inverted_p3(&joints, 10), y_inverted_p3(&joints, 11)),
        ])
    }
}

impl fmt::Display for HagglingSession {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "buyer_id\t\t={}
seller_ids\t\t={:?}
left_seller_id\t\t={}
right_seller_id\t\t={}
winner_id\t\t={}
start_frame\t\t={}
subject_0={}
subject_1={}
subject_2={}",
            self.buyer_id,
            self.seller_ids,
            self.left_seller_id,
            self.right_seller_id,
            self.winner_id,
            self.start_frame,
            self.subjects[0],
            self.subjects[1],
            self.subjects[2],
        )
    }
}

impl fmt::Display for HagglingSessionSubject {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "
    human_id\t\t={}
    start_frame\t\t={}
    b_valid\t\t={}
    #frames\t\t={}",
            self.human_id,
            self.start_frame,
            self.b_valid,
            self.frames.len(),
        )
    }
}
