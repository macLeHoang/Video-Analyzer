import numpy as np
from third_parties.byte_track.byte_tracker import STrack

def do_track_byte_track(byte_tracker, 
                        batch_metadatas,
                        img_info=None, # img_info, (height, width, frame_id, video_id, file_name)
                        img_size=None):
    """
    img_info and img_size are specified if needed only, 
    in update function the two arguments are used to define box scale
    """
    dets = np.concatenate([batch_metadatas.xyxy, 
                           batch_metadatas.confidence[..., None],
                           batch_metadatas.class_id[..., None]], axis=1)
    tracked_stracks = byte_tracker.update(dets, img_info, img_size)

    xyxy     = np.array([STrack.tlwh_to_tlbr(strack.tlwh) for strack in tracked_stracks]).astype("int")
    score    = np.array([strack.score for strack in tracked_stracks])
    cls_id   = np.array([strack.class_id for strack in tracked_stracks]).astype("int") 
    track_id = np.array([strack.track_id for strack in tracked_stracks])
    return (xyxy, score, cls_id, track_id)


# Tracker zoo
tracker_zoo = {
    "byte_track": do_track_byte_track
}