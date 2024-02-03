from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.config import get_cfg


import cv2
import random
import os
import glob

from detectron2.utils.env import TORCH_VERSION
from detectron2.export import dump_torchscript_IR, scripting_with_instances
from detectron2.structures import Boxes
from detectron2.modeling import GeneralizedRCNN
from detectron2.utils.file_io import PathManager

import os
import torch
from torch import Tensor, nn
from typing import Dict, List, Tuple

class Custom_Object_Trainer:

    def __init__(self, model='faster_rcnn_X_101_32x8d_FPN_3x'):
        train_dir = r'C:\Users\timot\Documents\DataScience\WeCloudData\Olyver_AI\olyver_ai\Barbell-small-2\train'
        valid_dir = r'C:\Users\timot\Documents\DataScience\WeCloudData\Olyver_AI\olyver_ai\Barbell-small-2\valid'
        test_dir = r'C:\Users\timot\Documents\DataScience\WeCloudData\Olyver_AI\olyver_ai\Barbell-small-2\test'
        #register_coco_instances(name of dataset, metadata(extra metadata, can be empty dict), json_file(annotation file), directory with images)
        register_coco_instances("barbell_train",{},f'{train_dir}/_annotations.coco.json', train_dir)
        register_coco_instances("barbell_val",{},f'{valid_dir}/_annotations.coco.json', valid_dir)
        register_coco_instances("barbell_test",{},f'{test_dir}/_annotations.coco.json', test_dir)
        self.cfg_obj = get_cfg() # obtain detectron2's default config
        # load values from a file specified under Detectron2's official configs/ directory
        self.cfg_obj.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{model}.yaml"))
        self.cfg_obj.DATASETS.TRAIN = ("barbell_train",)
        self.cfg_obj.DATASETS.TEST = ("barbell_val",)
        self.cfg_obj.DATALOADER.NUM_WORKERS = 0 # originally 4, set to 0 to disable parallel loads
        self.cfg_obj.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-Detection/{model}.yaml")
        self.cfg_obj.SOLVER.IMS_PER_BATCH = 4
        self.cfg_obj.SOLVER.BASE_LR = 0.001 # learning rate
        self.cfg_obj.SOLVER.WARMUP_ITERS = 1000
        self.cfg_obj.SOLVER.MAX_ITER = 1500 #adjust up if val mAP is still rising, adjust down if overfit
        # Solver steps changed end of range from 1500 to 1400 because it has to be lower than max iter i think
        self.cfg_obj.SOLVER.STEPS = (1000,1400)
        self.cfg_obj.SOLVER.GAMMA = 0.05
        self.cfg_obj.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
        self.cfg_obj.MODEL.ROI_HEADS.NUM_CLASSES = 2 # adjust for number of classes
        self.cfg_obj.TEST.EVAL_PERIOD = 500
        #self.cfg_obj.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
    def view_training_samples(self, num_sample=3):
        barbell_train_metadata = MetadataCatalog.get("barbell_train")
        dataset_dicts = DatasetCatalog.get("barbell_train")
        for d in random.sample(dataset_dicts, num_sample):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=barbell_train_metadata, scale=0.5)
            vis = visualizer.draw_dataset_dict(d)
            cv2.imshow('barbell',vis.get_image()[:, :, ::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    def train(self):
        # Need to make sure out model validates against our validation set
        class CocoTrainer(DefaultTrainer): #DefaultTrainer creates a model for training with the given config.

            @classmethod
            def build_evaluator(cls, cfg, dataset_name, output_folder=None): # trainer will look for this method

                if output_folder is None:
                    os.makedirs("coco_eval", exist_ok=True)
                    output_folder = "coco_eval"
                # Evaluate Mean Average Precision(mAP)
                return COCOEvaluator(dataset_name, cfg, False, output_folder) # evaluate on training/validation data
                # evaluator puts results in output folder. results are:
                #  1. "instances_predictions.pth" a file that can be loaded with `torch.load` and
                #    contains all the results in the format they are produced by the model.
                #  2. "coco_instances_results.json" a json file in COCO's result format.
        # os.makedirs(self.cfg_obj.OUTPUT_DIR, exist_ok=True)
        self.trainer = CocoTrainer(self.cfg_obj)
        self.trainer.resume_or_load(resume=False) #do not resume from checkpoint of model weights
        self.trainer.train() # trainer makes ./output folder and puts model weights there
        print('Training Complete')
        # Look at training curves in tensorboard by using Python:Launch Tensorboard in command palette and selecting coco_eval dir
        #Loss appeared to bottom out at 1400
    def evaluate(self):
        """ Evaluates trained model on the test dataset. """
        self.cfg_obj.MODEL.WEIGHTS = os.path.join(self.cfg_obj.OUTPUT_DIR, "model_final.pth")
        self.cfg_obj.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
        evaluator = COCOEvaluator("barbell_test", self.cfg_obj, False, output_dir="./output/") #evaluator on test data
        val_loader = build_detection_test_loader(self.cfg_obj, "barbell_test") # processes raw datasets into format needed by model
        metrics = inference_on_dataset(self.trainer.model, val_loader, evaluator) # (model, data_loader, evaluator)
        # runs model on the data_loader and evaluates metrics using the evaluator
        return metrics, self.trainer.model, self.cfg_obj.MODEL.WEIGHTS
    
    def view_testing_samples(self):
        predictor = DefaultPredictor(self.cfg_obj)
        barbell_metadata = MetadataCatalog.get("barbell_test")
        # this code is for showing all images in test folder with predictions drawn over it
        for imageName in glob.glob('/content/detectron2/test/*jpg'):
            im = cv2.imread(imageName)
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1],
                            metadata=barbell_metadata,
                            scale=0.8
                            )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imshow(out.get_image()[:, :, ::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    def export_scripting(self, output_name=str):
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
        model = self.trainer.model
        class ScriptableAdapterBase(nn.Module):
            # Use this adapter to workaround https://github.com/pytorch/pytorch/issues/46944
            # by not retuning instances but dicts. Otherwise the exported model is not deployable
            def __init__(self):
                super().__init__()
                self.model = model
                # self.model = torch_model
                self.eval()

        # if isinstance(torch_model, GeneralizedRCNN):
        if isinstance(model, GeneralizedRCNN):
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
            f"{output_name}.ts"), "wb") as f:
            torch.jit.save(ts_model, f)
        dump_torchscript_IR(ts_model, output_path)
        # TODO inference in Python now missing postprocessing glue code
        return None