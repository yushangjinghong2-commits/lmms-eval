import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Add FUDOKI repository to Python path
wd = Path(__file__).parent.parent.parent.parent.resolve()
fudoki_path = os.path.join(str(wd), "FUDOKI")
# Also try parent directory of lmms-eval
if not os.path.exists(fudoki_path):
    fudoki_path = os.path.join(str(wd.parent), "FUDOKI")
if os.path.exists(fudoki_path):
    sys.path.insert(0, fudoki_path)
    eval_logger.info(f"Added FUDOKI path to sys.path: {fudoki_path}")
else:
    eval_logger.warning(
        f"FUDOKI repository not found at {fudoki_path}. "
        f"Please clone it: git clone https://github.com/fudoki-hku/FUDOKI.git"
    )


# Constants from FUDOKI
VOCABULARY_SIZE_TXT = 102400
VOCABULARY_SIZE_IMG = 16384
IMG_LEN = 576


def resize_pad(image, image_size=384):
    """Resize and pad image to square with gray padding."""
    w, h = image.size
    if w <= 0 or h <= 0:
        return image.resize((image_size, image_size), Image.Resampling.BILINEAR)

    resize_scale = image_size / max(w, h)
    new_w = max(1, int(w * resize_scale))
    new_h = max(1, int(h * resize_scale))

    padding_color = (127, 127, 127)
    new_image = Image.new('RGB', (image_size, image_size), padding_color)

    if new_w <= 0 or new_h <= 0:
        return image.resize((image_size, image_size), Image.Resampling.BILINEAR)

    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    paste_x = (image_size - new_w) // 2
    paste_y = (image_size - new_h) // 2

    new_image.paste(image, (paste_x, paste_y))
    return new_image


@register_model("fudoki")
class FUDOKI(lmms):
    """
    FUDOKI: Discrete Flow-based Unified Understanding and Generation
    https://huggingface.co/LucasJinWang/FUDOKI
    https://github.com/fudoki-hku/FUDOKI
    """

    def __init__(
        self,
        pretrained: str = "LucasJinWang/FUDOKI",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "float32",
        batch_size: int = 1,
        discrete_fm_steps: int = 50,
        txt_max_length: int = 500,
        image_size: int = 384,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.pretrained = pretrained
        self.discrete_fm_steps = discrete_fm_steps
        self.txt_max_length = txt_max_length
        self.image_size = image_size

        # Setup accelerator
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device

        # Determine dtype
        if dtype == "bfloat16" or dtype == "bf16":
            self._dtype = torch.bfloat16
        elif dtype == "float16" or dtype == "fp16":
            self._dtype = torch.float16
        elif dtype == "float32" or dtype == "fp32":
            self._dtype = torch.float32
        else:
            self._dtype = torch.float32

        # Load model
        eval_logger.info(f"Loading FUDOKI model from {pretrained}")
        self._load_model(pretrained)

        self.batch_size_per_gpu = int(batch_size)
        assert self.batch_size_per_gpu == 1, "batch_size > 1 not supported for FUDOKI"

        # Setup distributed training
        if accelerator.num_processes > 1:
            distributed_type_list = [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ]
            assert accelerator.distributed_type in distributed_type_list, (
                "Unsupported distributed type. Only DDP, FSDP, and DeepSpeed supported"
            )
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self._model)
            else:
                self._model = accelerator.prepare_model(
                    self._model, evaluation_mode=True
                )
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(
                    f"Using {accelerator.num_processes} devices with parallelism"
                )
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    def _load_model(self, model_path: str) -> None:
        """Load FUDOKI model and processor."""
        try:
            from fudoki.model import instantiate_model
            from fudoki.janus.models import VLChatProcessor
            from fudoki.eval_loop import CFGScaledModel
            from flow_matching.path import MixtureDiscreteSoftmaxProbPath
            from flow_matching.solver import MixtureDiscreteSoftmaxEulerSolver
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            eval_logger.error(
                f"Failed to import FUDOKI modules. "
                f"Make sure FUDOKI repository is in the parent directory. Error: {e}"
            )
            raise

        eval_logger.info(f"Loading FUDOKI model from {model_path}")

        # Load model - supports both local path and HuggingFace model ID
        self._model = instantiate_model(model_path).to(self._device).to(self._dtype)
        self._model.train(False)
        self._model.eval()

        # Load processor
        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)

        # Setup CFG model for understanding
        self.cfg_weighted_model = CFGScaledModel(
            model=self._model, g_or_u='understanding'
        )

        # Setup flow matching paths and solver
        # Try to find embedding files locally first, then download from HuggingFace
        text_embedding_path = None
        image_embedding_path = None

        # Check if model_path is a local directory
        if os.path.isdir(model_path):
            local_text_emb = os.path.join(model_path, "text_embedding.pt")
            local_image_emb = os.path.join(model_path, "image_embedding.pt")
            if os.path.exists(local_text_emb):
                text_embedding_path = local_text_emb
            if os.path.exists(local_image_emb):
                image_embedding_path = local_image_emb

        # Download from HuggingFace if not found locally
        if text_embedding_path is None:
            eval_logger.info("Downloading text_embedding.pt from HuggingFace...")
            try:
                text_embedding_path = hf_hub_download(
                    repo_id=model_path if "/" in model_path else "LucasJinWang/FUDOKI",
                    filename="text_embedding.pt"
                )
                eval_logger.info(f"Downloaded text_embedding.pt to {text_embedding_path}")
            except Exception as e:
                eval_logger.error(f"Failed to download text_embedding.pt: {e}")
                raise

        if image_embedding_path is None:
            eval_logger.info("Downloading image_embedding.pt from HuggingFace...")
            try:
                image_embedding_path = hf_hub_download(
                    repo_id=model_path if "/" in model_path else "LucasJinWang/FUDOKI",
                    filename="image_embedding.pt"
                )
                eval_logger.info(f"Downloaded image_embedding.pt to {image_embedding_path}")
            except Exception as e:
                eval_logger.error(f"Failed to download image_embedding.pt: {e}")
                raise

        self.path_txt = MixtureDiscreteSoftmaxProbPath(
            mode='text', embedding_path=text_embedding_path
        )
        self.path_img = MixtureDiscreteSoftmaxProbPath(
            mode='image', embedding_path=image_embedding_path
        )
        self.solver = MixtureDiscreteSoftmaxEulerSolver(
            model=self.cfg_weighted_model,
            path_txt=self.path_txt,
            path_img=self.path_img,
            vocabulary_size_txt=VOCABULARY_SIZE_TXT,
            vocabulary_size_img=VOCABULARY_SIZE_IMG,
        )

        # Setup image transform
        self.image_transform = transforms.Compose([
            transforms.Lambda(lambda img: resize_pad(img, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            )
        ])

        eval_logger.info("FUDOKI model loaded successfully")

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self.vl_chat_processor.tokenizer

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self.vl_chat_processor.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.txt_max_length + IMG_LEN

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, **kwargs):
        return self.tokenizer.encode(string, **kwargs)

    def tok_decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens, **kwargs)

    def flatten(self, input):
        """Flatten nested lists."""
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        """Generate responses for a batch of requests."""
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Model Responding"
        )

        for request in requests:
            try:
                result = self._generate_single(request)
                res.append(result)
            except Exception as e:
                eval_logger.error(f"Error generating response: {e}")
                res.append("")
            pbar.update(1)

        pbar.close()
        return res

    def _generate_single(self, request: Instance) -> str:
        """Generate response for a single request."""
        # Extract request components - Instance is a tuple-like object
        # request.args is (context, generation_kwargs)
        context, generation_kwargs = request.args

        # Get visual input
        visuals = request.doc.get("visual", []) if hasattr(request, "doc") else []

        # Get question from context or doc
        if hasattr(request, "doc") and "question" in request.doc:
            question = request.doc["question"]
        else:
            # Extract question from context
            question = context if isinstance(context, str) else ""

        if not visuals:
            eval_logger.warning("No visual input provided")
            return ""

        # Process image
        image_path = visuals[0]
        if isinstance(image_path, str):
            img = Image.open(image_path).convert("RGB")
        elif isinstance(image_path, Image.Image):
            img = image_path.convert("RGB")
        else:
            eval_logger.error(f"Unsupported visual type: {type(image_path)}")
            return ""

        # Create conversation
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>{question}"
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]

        # Apply SFT template
        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt=self.vl_chat_processor.system_prompt,
        )

        # Process image if placeholder exists
        if '<image_placeholder>' in sft_format:
            img_tensor = self.image_transform(img)
            img_len = IMG_LEN
        else:
            img_tensor = None
            img_len = IMG_LEN

        # Tokenize
        input_ids = self.vl_chat_processor.tokenizer.encode(sft_format)
        input_ids = torch.LongTensor(input_ids)

        # Add image tokens
        image_token_mask = (input_ids == self.vl_chat_processor.image_id)
        image_indices = image_token_mask.nonzero()
        input_ids, _ = self.vl_chat_processor.add_image_token(
            image_indices=image_indices,
            input_ids=input_ids,
        )

        # Pad tokens
        original_input_id_len = input_ids.shape[0]
        if original_input_id_len >= self.txt_max_length + img_len:
            eval_logger.warning("Input too long, truncating")
            input_ids = input_ids[:self.txt_max_length + img_len]
        else:
            rows_to_pad = self.txt_max_length + img_len - input_ids.shape[0]
            input_ids = torch.cat([
                input_ids,
                torch.LongTensor([self.vl_chat_processor.pad_id]).repeat(rows_to_pad)
            ], dim=0)

        # Prepare batch
        input_ids = input_ids.unsqueeze(0).to(self._device)
        if img_tensor is not None:
            img_tensor = img_tensor.unsqueeze(0).to(self._device)

        # Generate with discrete flow matching
        with torch.no_grad():
            output = self.solver.sample(
                x_init=input_ids,
                img=img_tensor,
                steps=self.discrete_fm_steps,
                cfg_scale=1.0,
            )

        # Decode output
        output_ids = output[0].cpu().tolist()
        response = self.vl_chat_processor.tokenizer.decode(
            output_ids, skip_special_tokens=True
        )

        return response

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not implemented for FUDOKI."""
        raise NotImplementedError(
            "Loglikelihood is not supported for FUDOKI (discrete flow matching model)"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue generation."""
        raise NotImplementedError(
            "Multi-round dialogue is not yet implemented for FUDOKI"
        )

