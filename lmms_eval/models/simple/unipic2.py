import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from accelerate import Accelerator
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("unipic2")
class UniPic2(lmms):
    """
    UniPic2-MetaQuery Multimodal Model for Image Understanding

    Based on Qwen2.5-VL, supports visual understanding tasks.

    Example usage:
    accelerate launch -m lmms_eval \
        --model unipic2 \
        --model_args pretrained=/path/to/UniPic2-MetaQuery \
        --tasks illusionbench_arshia_icon_shape_test \
        --batch_size 1 \
        --output_path ./logs/
    """

    def __init__(
        self,
        pretrained: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        top_p: float = 1.0,
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        continual_mode: bool = True,
        response_persistent_folder: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        self.device_str = device
        self.continual_mode = continual_mode

        # Determine dtype
        if dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

        # Setup response cache for continual mode
        self.response_cache = {}
        self.cache_mode = "start"

        if response_persistent_folder is None:
            self.response_persistent_folder = "./logs/unipic2_persistent_folder"
        else:
            self.response_persistent_folder = response_persistent_folder

        if self.continual_mode:
            os.makedirs(self.response_persistent_folder, exist_ok=True)
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder, "unipic2_response.json"
            )

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file) as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
                eval_logger.info(
                    f"Loaded cache: {len(self.response_cache)} records"
                )

        # Setup accelerator
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            if self.continual_mode:
                eval_logger.warning(
                    "Continual mode is not supported for distributed inference. "
                    "Automatically disabling continual_mode."
                )
                self.continual_mode = False
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = 0
            self._world_size = 1

        # Load model
        eval_logger.info(f"Loading UniPic2 model from {pretrained}")
        self._load_model(attn_implementation)

        eval_logger.info("UniPic2 model initialized successfully")

    def _load_model(self, attn_implementation: str):
        """Load Qwen2.5-VL model and processor"""
        # Load processor
        self._processor = Qwen2_5_VLProcessor.from_pretrained(
            self.pretrained, trust_remote_code=True
        )

        # Modify chat template to remove system prompt
        self._processor.chat_template = self._processor.chat_template.replace(
            "{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}",
            "",
        )

        # Load model
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.pretrained,
            device_map="auto",
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        ).eval()

        eval_logger.info(
            f"Model loaded with dtype={self.torch_dtype}, "
            f"attn_implementation={attn_implementation}"
        )

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._processor.tokenizer

    @property
    def processor(self):
        return self._processor

    def understand_image(self, prompt: str, image: Image.Image) -> str:
        """
        Understand image and answer question

        Args:
            prompt: Input text prompt/question
            image: PIL Image to understand

        Returns:
            Generated text answer
        """
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(self.device_str)

        # Generate
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            top_p=self.top_p,
        )

        # Decode output
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Remove the input prompt from output if present
        if text in generated_text:
            generated_text = generated_text.replace(text, "").strip()

        return generated_text

    def flatten(self, input_list):
        """Flatten a nested list"""
        output = []
        for item in input_list:
            if isinstance(item, list):
                output.extend(self.flatten(item))
            else:
                output.append(item)
        return output

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Main inference method"""
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="UniPic2 Generating",
        )

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [
            reg.args for reg in requests
        ]:
            doc_uuid = get_uuid(task, split, doc_id)

            # Check cache
            if self.continual_mode and self.cache_mode == "resume":
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content:
                        res.append(content)
                        pbar.update(1)
                        continue

            prompt = contexts

            # Get image from doc_to_visual
            if doc_to_visual is None:
                eval_logger.warning(
                    f"No image provided for understanding mode, doc_id={doc_id}"
                )
                res.append("")
                pbar.update(1)
                continue

            # Get image from doc_to_visual
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            if not visuals or len(visuals) == 0:
                eval_logger.warning(f"No visual data found for doc_id={doc_id}")
                res.append("")
                pbar.update(1)
                continue

            # Use first image for understanding
            image = visuals[0]
            output_text = self.understand_image(prompt, image)

            res.append(output_text)

            # Update cache
            if self.continual_mode:
                self.response_cache[doc_uuid] = output_text
                with open(self.response_persistent_file, "w") as f:
                    json.dump(
                        self.response_cache, f, ensure_ascii=False, indent=2
                    )

            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError(
            "UniPic2 is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue generation"""
        raise NotImplementedError(
            "TODO: Implement multi-round dialogue generation"
        )
