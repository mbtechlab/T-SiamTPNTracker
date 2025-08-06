from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
# For debugging
import numpy as np
import torch.nn.functional as F
import cv2
import os
from lib.models.siamtpn_st.track_st import build_network
from lib.test.tracker.utils import Preprocessor
from lib.utils.box_ops import clip_box
from copy import deepcopy


class TSiamTPN(BaseTracker):
    def __init__(self, params):
        super(TSiamTPN, self).__init__(params)
        network = build_network(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
        self.cfg = params.cfg
        if params.cpu:
            self.network = network
        else:
            self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor(cpu=params.cpu)

        # Debug mode
        self.debug = self.params.debug
        self.frame_id = 0

        self.update_interval = self.cfg.TEST.UPDATE_INTERVALS.ANY
        print("Update interval:", self.update_interval)
        self.num_extra_template = 4  # In addition to the first template, there are 4 extra templates

        self.grids = self._generate_anchors(self.cfg.MODEL.ANCHOR.NUM, self.cfg.MODEL.ANCHOR.FACTOR, self.cfg.MODEL.ANCHOR.BIAS)
        self.window = self._hanning_window(self.cfg.MODEL.ANCHOR.NUM)
        self.hanning_factor = self.cfg.TEST.HANNING_FACTOR
        self.feat_sz_tar = self.cfg.MODEL.ANCHOR.NUM

        # Flags and previous states
        self.correction_done = False
        self.prev_state = None
        self.prev_templates = None

        # Normalization factor
        self.alpha = 0.9

        # Activation parameters
        self.enable_smoothing = getattr(params, 'enable_smoothing', True)
        self.enable_correction = getattr(params, 'enable_correction', False)

        print(f"Smoothing is {'enabled' if self.enable_smoothing else 'disabled'}.")
        print(f"Bounding box correction is {'enabled' if self.enable_correction else 'disabled'}.")

        # New variables to store the last stable frame
        self.last_stable_state = None
        self.last_stable_templates = None

    def initialize(self, image, info: dict):
        self.templates_list = []
        gt_box = torch.tensor(info['init_bbox']).float()

        # Extract the first template
        z_patch_arr1, _, z_amask_arr1 = sample_target(image, gt_box, self.params.template_factor,
                                                      output_sz=self.params.template_size)
        template1 = self.preprocessor.process(z_patch_arr1, z_amask_arr1)
        self.templates_list.append(template1)

        # Add 4 extra templates
        for _ in range(self.num_extra_template):
            template_copy = deepcopy(template1)
            self.templates_list.append(template_copy)

        self.fpn_outputs = []
        with torch.no_grad():
            for template in self.templates_list:
                tem_feat = self.network.backbone(template)
                tem_feat_fpn = self.network.fpn(tem_feat)
                self.fpn_outputs.append(tem_feat_fpn)

        self.state = info['init_bbox'].copy()
        self.frame_id = 0
        self.prev_state = self.state.copy()
        self.prev_templates = deepcopy(self.templates_list)
        
        # Initially, consider the first frame as stable
        self.last_stable_state = self.state.copy()
        self.last_stable_templates = deepcopy(self.templates_list)

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1

        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)

        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        combined_train_feats = torch.cat(self.fpn_outputs, dim=1)

        with torch.no_grad():
            tar_feat = self.network.backbone(search)
            tar_feat = self.network.fpn(tar_feat)

            outputs = self.network.head(tar_feat, combined_train_feats)
            raw_scores = outputs['pred_cls']
            boxes = outputs['pred_box']
            raw_scores = raw_scores.cpu()  
            boxes = boxes.cpu()
            pred_boxes = boxes.reshape(-1, 4)
            lt = self.grids[:, :2] - pred_boxes[:, :2]
            rb = self.grids[:, :2] + pred_boxes[:, 2:]
            pred_boxes = torch.cat([lt, rb], -1).view(-1, 4)

        raw_scores = torch.sigmoid(raw_scores).view(self.feat_sz_tar, self.feat_sz_tar)
        raw_scores = raw_scores * (1 - self.hanning_factor) + self.hanning_factor * self.window
        max_v, ind = raw_scores.view(-1).topk(1)  
        pred_box = pred_boxes[ind, :]

        pred_box = (pred_box.mean(dim=0) * self.params.search_size / resize_factor).tolist()
        current_state = self.state.copy()
        current_templates = deepcopy(self.templates_list)

        new_state = self.map_box_back(pred_box, resize_factor)
        new_state = clip_box(new_state, H, W, margin=10)

        if self.enable_smoothing and self.prev_state is not None:
            smoothed_state = [
                self.alpha * new_state[i] + (1 - self.alpha) * self.prev_state[i] for i in range(4)
            ]
            self.state = smoothed_state
        else:
            self.state = new_state

        # Calculate changes for correction
        if self.enable_correction and not self.correction_done and self.prev_state is not None:
            prev_cx = self.prev_state[0] + 0.5 * self.prev_state[2]
            prev_cy = self.prev_state[1] + 0.5 * self.prev_state[3]

            current_cx = self.state[0] + 0.5 * self.state[2]
            current_cy = self.state[1] + 0.5 * self.state[3]

            displacement = np.sqrt((current_cx - prev_cx) ** 2 + (current_cy - prev_cy) ** 2)
            prev_diagonal = np.sqrt(self.prev_state[2] ** 2 + self.prev_state[3] ** 2)
            size_change_ratio = (self.state[2] * self.state[3]) / (self.prev_state[2] * self.prev_state[3])

            displacement_threshold = 1.5 * prev_diagonal  
            size_change_threshold = 2.0

            # If thresholds are exceeded, apply correction
            if displacement > displacement_threshold or size_change_ratio > size_change_threshold:
                print(f"Correction activated at frame {self.frame_id}. Displacement: {displacement:.2f}, Size change ratio: {size_change_ratio:.2f}")

                # Use the last stable frame if available
                if self.last_stable_templates is not None and self.last_stable_state is not None:
                    self.state = self.last_stable_state.copy()
                    self.templates_list = [self.templates_list[0]] + deepcopy(self.last_stable_templates[1:])
                else:
                    # Otherwise, use the previous state
                    self.state = self.prev_state.copy()
                    self.templates_list = [self.templates_list[0]] + deepcopy(self.prev_templates[1:])

                self.fpn_outputs = []
                with torch.no_grad():
                    for template in self.templates_list:
                        tem_feat = self.network.backbone(template)
                        tem_feat_fpn = self.network.fpn(tem_feat)
                        self.fpn_outputs.append(tem_feat_fpn)

                self.correction_done = True
            else:
                # If no correction is needed, save this frame as stable
                self.last_stable_state = self.state.copy()
                self.last_stable_templates = deepcopy(self.templates_list)

        # Save the current state for the next frame
        self.prev_state = self.state.copy()
        self.prev_templates = deepcopy(self.templates_list)

        conf_score = max_v.item()
        if self.frame_id % self.update_interval == 0 and conf_score > 0.75:
            z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                        output_sz=self.params.template_size)
            template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
            with torch.no_grad():
                tem_feat_t = self.network.backbone(template_t)
                tem_feat_fpn_t = self.network.fpn(tem_feat_t)

            self.templates_list.pop(1)
            self.fpn_outputs.pop(1)

            self.templates_list.append(template_t)
            self.fpn_outputs.append(tem_feat_fpn_t)

            self.templates_list = [self.templates_list[0]] + self.templates_list[1:]
            self.fpn_outputs = [self.fpn_outputs[0]] + self.fpn_outputs[1:]

            print(f"Template updated at frame {self.frame_id} with conf_score: {conf_score}")

        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            cv2.imshow("demo", image_BGR)
            key = cv2.waitKey(1)
            if key == ord('p'):
                cv2.waitKey(-1)

        return {"target_bbox": self.state,
                "conf_score": conf_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        x1, y1, x2, y2 = pred_box
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def _hanning_window(self, num):
        hanning = np.hanning(num)
        window = np.outer(hanning, hanning)
        window = torch.from_numpy(window)
        return window

    def _generate_anchors(self, num=20, factor=1, bias=0.5):
        x = np.arange(num)
        y = np.arange(num)
        xx, yy = np.meshgrid(x, y)
        xx = (factor * xx + bias) / num
        yy = (factor * yy + bias) / num
        xx = torch.from_numpy(xx).view(-1).float()
        yy = torch.from_numpy(yy).view(-1).float()
        grids = torch.stack([xx, yy], -1)
        return grids


def get_tracker_class():
    return TSiamTPN
