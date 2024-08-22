import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig
from evf_sam.model.segment_anything.utils.transforms import ResizeLongestSide
from evf_sam.model.evf_sam2 import EvfSam2Model

class EVFInference:
    def __init__(self, version, vis_save_path="./infer", precision="fp16", image_size=224, model_max_length=512, local_rank=0, load_in_8bit=False, load_in_4bit=False, model_type="ori", image_path="./assets/zebra.jpg", prompt="zebra top left"):
        self.version = version
        self.vis_save_path = vis_save_path
        self.precision = precision
        self.image_size = image_size
        self.model_max_length = model_max_length
        self.local_rank = local_rank
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.model_type = model_type
        self.image_path = image_path
        self.prompt = prompt
        self.tokenizer, self.model = self.init_models()

    def init_models(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.version,
            padding_side="right",
            use_fast=False,
        )

        torch_dtype = torch.float32
        if self.precision == "bf16":
            torch_dtype = torch.bfloat16
        elif self.precision == "fp16":
            torch_dtype = torch.half

        kwargs = {"torch_dtype": torch_dtype}
        if self.load_in_4bit:
            kwargs.update(
                {
                    "torch_dtype": torch.half,
                    "quantization_config": BitsAndBytesConfig(
                        llm_int8_skip_modules=["visual_model"],
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    ),
                }
            )
        elif self.load_in_8bit:
            kwargs.update(
                {
                    "torch_dtype": torch.half,
                    "quantization_config": BitsAndBytesConfig(
                        llm_int8_skip_modules=["visual_model"],
                        load_in_8bit=True,
                    ),
                }
            )

        model = EvfSam2Model.from_pretrained(self.version, low_cpu_mem_usage=True, **kwargs)

        if (not self.load_in_4bit) and (not self.load_in_8bit):
            model = model.cuda()
        model.eval()

        return tokenizer, model

    def preprocess_image(self, image_path):
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        image_beit = self.beit3_preprocess(image_np, self.image_size).to(dtype=self.model.dtype, device=self.model.device)

        image_sam, resize_shape = self.sam_preprocess(image_np)
        image_sam = image_sam.to(dtype=self.model.dtype, device=self.model.device)

        return image_sam, image_beit, resize_shape, original_size_list

    def sam_preprocess(
        self,
        x: np.ndarray,
        pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
        pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
        img_size=1024) -> torch.Tensor:
        '''
        preprocess of Segment Anything Model, including scaling, normalization and padding.  
        preprocess differs between SAM and Effi-SAM, where Effi-SAM use no padding.
        input: ndarray
        output: torch.Tensor
        '''
        assert img_size==1024, \
            "both SAM and Effi-SAM receive images of size 1024^2, don't change this setting unless you're sure that your employed model works well with another size."
        x = ResizeLongestSide(img_size).apply_image(x)
        resize_shape = x.shape[:2]
        x = torch.from_numpy(x).permute(2,0,1).contiguous()

        # Normalize colors
        x = (x - pixel_mean) / pixel_std
        x = F.interpolate(x.unsqueeze(0), (img_size, img_size), mode="bilinear").squeeze(0)
        
        return x, resize_shape

    def beit3_preprocess(self, x: np.ndarray, img_size=224) -> torch.Tensor:
        '''
        preprocess for BEIT-3 model.
        input: ndarray
        output: torch.Tensor
        '''
        beit_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC), 
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        return beit_preprocess(x)

    def infer(self, image_sam, image_beit, resize_shape, original_size_list):
        input_ids = self.tokenizer(self.prompt, return_tensors="pt")["input_ids"].to(device=self.model.device)
        pred_mask = self.model.inference(
            image_sam.unsqueeze(0),
            image_beit.unsqueeze(0),
            input_ids,
            resize_list=[resize_shape],
            original_size_list=original_size_list,
        )
        return pred_mask.detach().cpu().numpy()[0] > 0

    def save_visualization(self, image_np, pred_mask, save_path):
        save_img = image_np.copy()
        save_img[pred_mask] = (
            image_np * 0.5
            + pred_mask[:, :, None].astype(np.uint8) * np.array([50, 120, 220]) * 0.5
        )[pred_mask]
        save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, save_img)

    def run_inference(self):
        if not os.path.exists(self.image_path):
            print("File not found in {}".format(self.image_path))
            return

        os.makedirs(self.vis_save_path, exist_ok=True)
        save_path = "{}/{}_vis.png".format(
            self.vis_save_path, os.path.basename(self.image_path).split(".")[0]
        )

        image_sam, image_beit, resize_shape, original_size_list = self.preprocess_image(self.image_path)
        pred_mask = self.infer(image_sam, image_beit, resize_shape, original_size_list)
        self.save_visualization(cv2.imread(self.image_path), pred_mask, save_path)

# Demo usage
if __name__ == "__main__":
    evf_infer = EVFInference('./evf-sam2')
    evf_infer.run_inference()