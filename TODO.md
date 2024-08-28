# Video to video

```py
@dataclass
class VideoPoint:
    x: float
    y: float
    frame: int

@dataclass
class Label:
    points: VideoPoint
    object_id: int
    label: bool

@dataclass
class Mask:
    # A mask in JPEG format
    # Each value in 0-225 is a different layer
    # 0 means no mask
    mask: bytes 

class Client:
    def __init__(self):
        self.state = None
        self.video_frames_dir = None
        self.inference_state = None

        # Hard stuff
        from sam2.build_sam import build_sam2_video_predictor

        sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"

        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    @api.post('/add-video')
    def add_video(
            self,
            video: bytes
        ):
        # video bytes to path
        video_path = Path('/tmp/file.mp4')
        video_path.write_bytes(io.BytesIO(video))

        # Get video frames dir
        video_frames_dir = Path('/tmp/frames/')
        os.subprocess.run(f"ffmpeg -i {video_path} -q:v 2 -start_number 0 {video_frames_dir}/'%05d.jpg'")
        self.video_frames_dir = video_frames_dir

        # SLOW: Needs progbar
        self.inference_state = predictor.init_state(video_path = video_frames_dir)

    @api.post('/reset')
    def reset(self):
        self.inference_state = predictor.init_state(video_path=self.video_frames_dir)

    @api.post('/click')
    def click(
        self,
        masks: list[Label]  # This gets accumulated in UI, and sent fully. 
    ) -> Mask:
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=ann_frame_idx, # mask.point.frame
            obj_id=ann_obj_id, # mask.object_id
            points=points, # From mask.point x y
            labels=labels, # From mask.label
        )
        mask = show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

        # UI receives this and overlays it on 
        return Mask(
            mask=to_jpeg_bytes(mask)
        )
    


```