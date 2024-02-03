import json
import torch
from detectron2.structures import Instances, Boxes

def json_to_d2(pred_dict, device):
    """ 
    Client side helper function to deserialize the JSON msg back to d2 outputs 
    """
    
    pred_dict = json.loads(pred_dict)
    for k, v in pred_dict.items():
        if k=="pred_boxes":
            boxes_to_tensor = torch.FloatTensor(v).to(device)
            pred_dict[k] = Boxes(boxes_to_tensor)
        if k=="scores":
            pred_dict[k] = torch.Tensor(v).to(device)
        if k=="pred_classes":
            pred_dict[k] = torch.Tensor(v).to(device).to(torch.uint8)
    
    height, width = pred_dict['image_size']
    del pred_dict['image_size']

    inst = Instances((height, width,), **pred_dict)
    
    return {'instances':inst}