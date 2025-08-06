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

        # For debugging
        self.debug = self.params.debug
        self.frame_id = 0

        # Update interval configuration
        self.update_interval = self.cfg.TEST.UPDATE_INTERVALS.ANY
        print("Update Interval:", self.update_interval)
        self.num_extra_template = 4  # In addition to the first template, we have 4 additional templates

        self.grids = self._generate_anchors(self.cfg.MODEL.ANCHOR.NUM, self.cfg.MODEL.ANCHOR.FACTOR, self.cfg.MODEL.ANCHOR.BIAS)
        self.window = self._hanning_window(self.cfg.MODEL.ANCHOR.NUM)
        self.hanning_factor = self.cfg.TEST.HANNING_FACTOR
        self.feat_sz_tar = self.cfg.MODEL.ANCHOR.NUM

        # Adding new variables for managing bounding box correction
        self.correction_done = False  # Flag to ensure the correction is applied only once
        self.prev_state = None  # To store the previous bounding box
        self.prev_templates = None  # To store the previous templates

        # Normalization factor for bounding box coordinates (0 < alpha <= 1)
        self.alpha = 0.9  # You can adjust this value between 0 (no smoothing) to 1 (full smoothing)

        # Adding parameters to enable/disable features
        self.enable_smoothing = getattr(params, 'enable_smoothing', False)  # Enabled by default
        self.enable_correction = getattr(params, 'enable_correction', True)  # Enabled by default

        print(f"Smoothing is {'enabled' if self.enable_smoothing else 'disabled'}.")
        print(f"Bounding box jump prevention is {'enabled' if self.enable_correction else 'disabled'}.")

    def initialize(self, image, info: dict):
        # List to store initial templates (as a class variable for later use)
        self.templates_list = []

        # Retrieve the initial bounding box
        gt_box = torch.tensor(info['init_bbox']).float()  # [x1, y1, w, h]

        # Extract the first template
        z_patch_arr1, _, z_amask_arr1 = sample_target(image, gt_box, self.params.template_factor,
                                                      output_sz=self.params.template_size)
        template1 = self.preprocessor.process(z_patch_arr1, z_amask_arr1)
        self.templates_list.append(template1)

        # Create copies of the first template and add them to the list
        for _ in range(self.num_extra_template):  # We already have template1, so we need 4 more copies
            template_copy = deepcopy(template1)
            self.templates_list.append(template_copy)

        # Initialize FPN outputs after passing through the backbone
        self.fpn_outputs = []  # Store as a class variable

        # Pass the templates through the backbone and FPN and store the outputs
        with torch.no_grad():
            for template in self.templates_list:
                tem_feat = self.network.backbone(template)  # Pass through the backbone
                tem_feat_fpn = self.network.fpn(tem_feat)  # Pass through the FPN
                self.fpn_outputs.append(tem_feat_fpn)  # Store the output in the class variable

        # Store the initial state (bounding box) and frame counter
        self.state = info['init_bbox'].copy()  # [x1, y1, w, h]
        self.frame_id = 0

        # Initialize the previous state and templates
        self.prev_state = self.state.copy()
        self.prev_templates = deepcopy(self.templates_list)

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1

        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)

        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        # Combine outputs using torch.cat along the channel dimension (dim=1)
        combined_train_feats = torch.cat(self.fpn_outputs, dim=1)

        with torch.no_grad():
            tar_feat = self.network.backbone(search)
            tar_feat = self.network.fpn(tar_feat)

            outputs = self.network.head(tar_feat, combined_train_feats)
            raw_scores = outputs['pred_cls']
            boxes = outputs['pred_box']
            raw_scores = raw_scores.cpu()  # B,L,2
            boxes = boxes.cpu()
            pred_boxes = boxes.reshape(-1, 4)
            lt = self.grids[:, :2] - pred_boxes[:, :2]
            rb = self.grids[:, :2] + pred_boxes[:, 2:]
            pred_boxes = torch.cat([lt, rb], -1).view(-1, 4)

        # Apply sigmoid to raw_scores and reshape to match feature map size
        raw_scores = torch.sigmoid(raw_scores).view(self.feat_sz_tar, self.feat_sz_tar)

        # Apply Hanning window adjustment to the scores
        raw_scores = raw_scores * (1 - self.hanning_factor) + self.hanning_factor * self.window

        # Find the maximum value and its index from the flattened raw_scores
        max_v, ind = raw_scores.view(-1).topk(1)  # Flatten scores and get the highest value

        # Select the predicted box corresponding to the highest score
        pred_box = pred_boxes[ind, :]

        pred_box = (pred_box.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]

        # Map heatmap back to the original image coordinates
        heatmap = raw_scores.numpy()  # Raw heatmap
        full_heatmap = self.map_heatmap_back(heatmap, resize_factor, (H, W))
        # Normalize the final heatmap
        if np.max(full_heatmap) > 0:
            heatmap_normalized = np.uint8(255 * full_heatmap / np.max(full_heatmap))  # Normalize to [0, 255]
        else:
            heatmap_normalized = np.zeros_like(full_heatmap, dtype=np.uint8)

        # Convert the heatmap to color
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

        # Combine the colored heatmap with the original image
        image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert image to BGR for OpenCV
        overlay = cv2.addWeighted(image_BGR, 0.6, heatmap_colored, 0.4, 0)

        # Save the combined image
        output_dir = "/home/ardi/Desktop/project/T-SiamTPNTracker/results/tracking_results/map/heatmap"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"frame_{self.frame_id:04d}.png")
        cv2.imwrite(output_path, overlay)  # Save the combined image

        # Before updating, store the current bounding box and templates
        current_state = self.state.copy()
        current_templates = deepcopy(self.templates_list)

        # Get the final bounding box result
        new_state = self.map_box_back(pred_box, resize_factor)
        new_state = clip_box(new_state, H, W, margin=10)  #

        # Apply smoothing to bounding box coordinates if enabled
        if self.enable_smoothing and self.prev_state is not None:
            smoothed_state = [
                self.alpha * new_state[i] + (1 - self.alpha) * self.prev_state[i] for i in range(4)
            ]
            self.state = smoothed_state
        else:
            self.state = new_state

        # Check bounding box changes per frame if jump prevention is enabled
        if self.enable_correction and not self.correction_done and self.prev_state is not None:
            # Compute the center of the previous bounding box
            prev_cx = self.prev_state[0] + 0.5 * self.prev_state[2]
            prev_cy = self.prev_state[1] + 0.5 * self.prev_state[3]

            # Compute the center of the current bounding box
            current_cx = self.state[0] + 0.5 * self.state[2]
            current_cy = self.state[1] + 0.5 * self.state[3]

            # Compute the displacement of the bounding box center
            displacement = np.sqrt((current_cx - prev_cx) ** 2 + (current_cy - prev_cy) ** 2)

            # Compute the diagonal of the previous bounding box
            prev_diagonal = np.sqrt(self.prev_state[2] ** 2 + self.prev_state[3] ** 2)

            # Compute the size change ratio
            size_change_ratio = (self.state[2] * self.state[3]) / (self.prev_state[2] * self.prev_state[3])

            # Define thresholds
            displacement_threshold = 1.5 * prev_diagonal  # Displacement greater than the previous bounding box diagonal
            size_change_threshold = 2.0  # Current bounding box size is more than 2x the previous bounding box

            # Check conditions
            if displacement > displacement_threshold or size_change_ratio > size_change_threshold:
                print(f"Correction activated at frame {self.frame_id}. Displacement: {displacement:.2f}, Size change ratio: {size_change_ratio:.2f}")

                # Replace the current bounding box with the previous bounding box
                self.state = self.prev_state.copy()

                # Replace templates except the first one with the previous templates
                self.templates_list = [self.templates_list[0]] + deepcopy(self.prev_templates[1:])

                # Update FPN outputs with the new templates
                self.fpn_outputs = []
                with torch.no_grad():
                    for template in self.templates_list:
                        tem_feat = self.network.backbone(template)
                        tem_feat_fpn = self.network.fpn(tem_feat)
                        self.fpn_outputs.append(tem_feat_fpn)

                # Set the flag to ensure this process is done only once
                self.correction_done = True

        # Store the current bounding box and templates as the previous for the next frame
        self.prev_state = self.state.copy()
        self.prev_templates = deepcopy(self.templates_list)

        # Update template
        conf_score = max_v.item()
        if self.frame_id % self.update_interval == 0 and conf_score > 0.75:
            # Sample a new template
            z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                        output_sz=self.params.template_size)
            template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
            with torch.no_grad():
                tem_feat_t = self.network.backbone(template_t)
                tem_feat_fpn_t = self.network.fpn(tem_feat_t)

            # Remove the second template
            self.templates_list.pop(1)
            self.fpn_outputs.pop(1)

            # Add the new template to the end (template5)
            self.templates_list.append(template_t)
            self.fpn_outputs.append(tem_feat_fpn_t)

            # Shift the templates (4 -> 3, 3 -> 2, etc.)
            self.templates_list = [self.templates_list[0]] + self.templates_list[1:]
            self.fpn_outputs = [self.fpn_outputs[0]] + self.fpn_outputs[1:]

            print(f"Updated template5 at frame {self.frame_id} with conf_score: {conf_score}")

        # For debugging
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

    # Other class methods remain unchanged

    def map_box_back(self, pred_box: list, resize_factor: float):
        # print(self.state)
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        x1, y1, x2, y2 = pred_box
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]
    
    def map_heatmap_back(self, heatmap: np.ndarray, resize_factor: float, image_size: tuple):
        """
        Map resized heatmap to the original image coordinates.
        
        Args:
            heatmap (np.ndarray): Raw heatmap.
            resize_factor (float): Resize factor.
            image_size (tuple): Original image dimensions (H, W).
        
        Returns:
            np.ndarray: Full heatmap with original image dimensions.
        """
        H, W = image_size
        search_size = self.params.search_size

        # Compute half the side of the search box in the original image
        half_side = 0.5 * search_size / resize_factor

        # Center of the previous bounding box
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]

        # Position of the search box in the original image
        x1 = int(cx_prev - half_side)
        y1 = int(cy_prev - half_side)
        x2 = int(cx_prev + half_side)
        y2 = int(cy_prev + half_side)

        # Ensure positions are within the image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)

        # Final resized heatmap size in the original image
        heatmap_height = y2 - y1
        heatmap_width = x2 - x1

        # Resize the heatmap to match the search box size in the original image
        heatmap_resized = cv2.resize(heatmap, (heatmap_width, heatmap_height), interpolation=cv2.INTER_LINEAR)

        # Create the full heatmap with original image dimensions
        full_heatmap = np.zeros((H, W), dtype=np.float32)

        # Place the resized heatmap at the correct position
        full_heatmap[y1:y2, x1:x2] = heatmap_resized

        return full_heatmap

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
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
        """
        Generate anchors for each sampled point.
        """
        x = np.arange(num)
        y = np.arange(num)
        xx, yy = np.meshgrid(x, y)
        xx = (factor * xx + bias) / num
        yy = (factor * yy + bias) / num
        xx = torch.from_numpy(xx).view(-1).float()
        yy = torch.from_numpy(yy).view(-1).float()
        grids = torch.stack([xx, yy], -1)  # N 2
        return grids


def get_tracker_class():
    return TSiamTPN
