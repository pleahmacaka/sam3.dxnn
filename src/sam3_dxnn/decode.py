import numpy as np
import torch
from transformers import Sam3Model, Sam3Processor
from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput


def load_sam3(model: str) -> tuple[Sam3Model, Sam3Processor]:
    return (
        Sam3Model.from_pretrained(model, dtype=torch.float32).eval(),
        Sam3Processor.from_pretrained(model),
    )


def vision_output(model: Sam3Model, hidden: np.ndarray, size: int) -> Sam3VisionEncoderOutput:
    states = torch.from_numpy(hidden).float()
    grid = size // model.config.vision_config.backbone_config.patch_size
    spatial = states.view(1, grid, grid, -1).permute(0, 3, 1, 2)
    fpn, position = model.vision_encoder.neck(spatial)

    return Sam3VisionEncoderOutput(
        last_hidden_state=states,
        fpn_hidden_states=fpn,
        fpn_position_encoding=position,
    )


def masks_of(
    model: Sam3Model,
    processor: Sam3Processor,
    hidden: np.ndarray,
    prompt: str,
    size: int,
    threshold: float,
) -> tuple[list[np.ndarray], list[float]]:
    text = processor(text=prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(
            vision_embeds=vision_output(model, hidden, size),
            input_ids=text["input_ids"],
            attention_mask=text["attention_mask"],
        )

    found = processor.post_process_instance_segmentation(outputs, threshold, 0.5, [(size, size)])[0]

    return [m.cpu().numpy().astype(bool) for m in found["masks"]], [float(s) for s in found["scores"]]
