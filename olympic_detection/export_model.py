from detectron2.utils.env import TORCH_VERSION
from detectron2.export import dump_torchscript_IR, scripting_with_instances
from detectron2.structures import Boxes
from detectron2.modeling import GeneralizedRCNN
from detectron2.utils.file_io import PathManager

import os
import torch
from torch import Tensor, nn
from typing import Dict, List, Tuple


def export_scripting(torch_model, output_name=str):
    assert TORCH_VERSION >= (1, 8)
    fields = {
        "proposal_boxes": Boxes,
        "objectness_logits": Tensor,
        "pred_boxes": Boxes,
        "scores": Tensor,
        "pred_classes": Tensor,
        "pred_masks": Tensor,
        "pred_keypoints": torch.Tensor,
        "pred_keypoint_heatmaps": torch.Tensor,
    }
   
    class ScriptableAdapterBase(nn.Module):
        # Use this adapter to workaround https://github.com/pytorch/pytorch/issues/46944
        # by not retuning instances but dicts. Otherwise the exported model is not deployable
        def __init__(self):
            super().__init__()
            self.model = torch_model
            self.eval()

    if isinstance(torch_model, GeneralizedRCNN):

        class ScriptableAdapter(ScriptableAdapterBase):
            def forward(self, inputs: Tuple[Dict[str, torch.Tensor]]) -> List[Dict[str, Tensor]]:
                instances = self.model.inference(inputs, do_postprocess=False)
                return [i.get_fields() for i in instances]

    else:

        class ScriptableAdapter(ScriptableAdapterBase):
            def forward(self, inputs: Tuple[Dict[str, torch.Tensor]]) -> List[Dict[str, Tensor]]:
                instances = self.model(inputs)
                return [i.get_fields() for i in instances]

    ts_model = scripting_with_instances(ScriptableAdapter(), fields)
    output_path = './models_for_deployment'
    os.makedirs('./models_for_deployment', exist_ok=True)
    # output_path = r'C:\Users\timot\Documents\DataScience\WeCloudData\Olyver_AI\olyver_ai\models_for_deployment'
    with PathManager.open(os.path.join(
        output_path,
        "model.ts"), "wb") as f:
        torch.jit.save(ts_model, f)
    dump_torchscript_IR(ts_model, output_path)
    # TODO inference in Python now missing postprocessing glue code
    return None