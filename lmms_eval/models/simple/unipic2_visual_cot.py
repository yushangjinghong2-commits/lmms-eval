"""
UniPic2 Visual Chain-of-Thought Model

Two-stage inference:
1. Stage 1: Generate visualization image from text prompt (using SD3.5M-Kontext)
2. Stage 2: Answer question using the generated image and original image (using Qwen2.5-VL)

Usage:
    accelerate launch -m lmms_eval \
        --model unipic2_visual_cot \
        --model_args pretrained=/path/to/UniPic2-SD3.5M-Kontext,qwen_model=/path/to/Qwen2.5-VL-7B-Instruct \
        --tasks illusionbench_arshia_icon_shape_visual_cot \
        --batch_size 1 \
        --device cuda:0 \
        --output_path ./logs/
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    T5EncoderModel,
    T5TokenizerFast,
)

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Add UniPic-2 path to sys.path to import custom modules
# Expected: lmms-eval/../UniPic/UniPic-2/ directory
wd = Path(__file__).parent.parent.parent.parent.resolve()
unipic2_path = os.path.join(str(wd), "UniPic", "UniPic-2")
if os.path.exists(unipic2_path):
    sys.path.append(unipic2_path)
    eval_logger.info(f"Added UniPic-2 path to sys.path: {unipic2_path}")
else:
    eval_logger.warning(
        f"UniPic-2 repository not found at {unipic2_path}. "
        f"Please clone it or ensure it's in the correct location."
    )


@register_model("unipic2_visual_cot")
class UniPic2VisualCoT(lmms):
    """
    UniPic2 Visual Chain-of-Thought Model

    Performs two-stage visual reasoning:
    1. Generate visualization image from text prompt (using SD3.5M-Kontext)
    2. Answer question using the generated image (using Qwen2.5-VL)
    """

    def __init__(
        self,
        pretrained: str,
        qwen_model: str,
        # Stage 1: Image generation parameters
        stage1_num_inference_steps: int = 50,
        stage1_guidance_scale: float = 3.5,
        stage1_height: int = 1024,
        stage1_width: int = 1024,
        # Stage 2: Visual understanding parameters
        stage2_max_new_tokens: int = 512,
        stage2_temperature: float = 0.0,
        stage2_do_sample: bool = False,
        stage2_top_p: float = 1.0,
        # Generation prompt template
        generation_prompt_template: str = "Generate a detailed visual diagram or illustration to help answer this question: {question}",
        # Output and debugging
        output_dir: Optional[str] = None,
        save_intermediate: bool = False,
        intermediate_dir: Optional[str] = None,
        # Error handling
        fail_gracefully: bool = True,
        # Model dtype
        dtype: str = "bfloat16",
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.qwen_model = qwen_model
        self.save_intermediate = save_intermediate
        self.fail_gracefully = fail_gracefully
        self.seed = seed
        self.generation_prompt_template = generation_prompt_template

        # Stage 1 parameters
        self.stage1_num_inference_steps = stage1_num_inference_steps
        self.stage1_guidance_scale = stage1_guidance_scale
        self.stage1_height = stage1_height
        self.stage1_width = stage1_width

        # Stage 2 parameters
        self.stage2_max_new_tokens = stage2_max_new_tokens
        self.stage2_temperature = stage2_temperature
        self.stage2_do_sample = stage2_do_sample
        self.stage2_top_p = stage2_top_p

        # Determine dtype
        if dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

        # Setup output directories
        if output_dir is None:
            self.output_dir = "./logs/unipic2_visual_cot"
        else:
            self.output_dir = output_dir

        if intermediate_dir is None:
            self.intermediate_dir = self.output_dir
        else:
            self.intermediate_dir = intermediate_dir

        if save_intermediate:
            os.makedirs(self.intermediate_dir, exist_ok=True)
            eval_logger.info(
                f"Intermediate artifacts will be saved under: {self.intermediate_dir}"
            )

        # Load models
        eval_logger.info(
            f"Loading UniPic2 SD3.5M-Kontext from {pretrained} and Qwen2.5-VL from {qwen_model}"
        )
        self._load_models()

        # Setup rank and world size
        self._rank = 0
        self._world_size = 1

        eval_logger.info("UniPic2VisualCoT initialized successfully")

    def _load_models(self):
        """Load SD3.5M-Kontext pipeline and Qwen2.5-VL model"""
        try:
            # Import UniPic2 custom modules
            from unipicv2.pipeline_stable_diffusion_3_kontext import (
                StableDiffusion3KontextPipeline,
            )
            from unipicv2.transformer_sd3_kontext import (
                SD3Transformer2DKontextModel,
            )

            # Load SD3.5M-Kontext components
            eval_logger.info("Loading SD3.5M-Kontext components...")

            transformer = SD3Transformer2DKontextModel.from_pretrained(
                self.pretrained,
                subfolder="transformer",
                torch_dtype=self.torch_dtype,
            ).cuda()

            vae = AutoencoderKL.from_pretrained(
                self.pretrained, subfolder="vae", torch_dtype=self.torch_dtype
            ).cuda()

            text_encoder = CLIPTextModelWithProjection.from_pretrained(
                self.pretrained,
                subfolder="text_encoder",
                torch_dtype=self.torch_dtype,
            ).cuda()
            tokenizer = CLIPTokenizer.from_pretrained(
                self.pretrained, subfolder="tokenizer"
            )

            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                self.pretrained,
                subfolder="text_encoder_2",
                torch_dtype=self.torch_dtype,
            ).cuda()
            tokenizer_2 = CLIPTokenizer.from_pretrained(
                self.pretrained, subfolder="tokenizer_2"
            )

            text_encoder_3 = T5EncoderModel.from_pretrained(
                self.pretrained,
                subfolder="text_encoder_3",
                torch_dtype=self.torch_dtype,
            ).cuda()
            tokenizer_3 = T5TokenizerFast.from_pretrained(
                self.pretrained, subfolder="tokenizer_3"
            )

            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                self.pretrained, subfolder="scheduler"
            )

            # Create SD3.5M-Kontext pipeline
            self.sd_pipeline = StableDiffusion3KontextPipeline(
                transformer=transformer,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                text_encoder_2=text_encoder_2,
                tokenizer_2=tokenizer_2,
                text_encoder_3=text_encoder_3,
                tokenizer_3=tokenizer_3,
                scheduler=scheduler,
            )

            eval_logger.info("SD3.5M-Kontext pipeline loaded successfully")

            # Load Qwen2.5-VL model
            eval_logger.info("Loading Qwen2.5-VL model...")
            self.qwen_processor = Qwen2_5_VLProcessor.from_pretrained(
                self.qwen_model, trust_remote_code=True
            )

            # Modify chat template to remove system prompt
            self.qwen_processor.chat_template = (
                self.qwen_processor.chat_template.replace(
                    "{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}",
                    "",
                )
            )

            self.qwen_model_instance = (
                Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.qwen_model,
                    device_map="auto",
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                ).eval()
            )

            eval_logger.info("Qwen2.5-VL model loaded successfully")

        except Exception as e:
            raise ImportError(
                f"Failed to import UniPic2 dependencies or load models. "
                f"Please ensure:\n"
                f"  1. UniPic-2 repository is available at {unipic2_path}\n"
                f"  2. Model weights are downloaded\n"
                f"Error: {e}"
            )

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def model(self):
        return self.qwen_model_instance

    @property
    def tokenizer(self):
        return self.qwen_processor.tokenizer

    def _stage1_generate_image(
        self, generation_prompt: str, doc_id: str, task: str
    ) -> Tuple[List[str]]:
        """
        Stage 1: Generate visualization image from prompt

        Args:
            generation_prompt: Text prompt for image generation
            doc_id: Document ID for file naming
            task: Task name for file naming

        Returns:
            List of image paths
        """
        eval_logger.debug(f"Stage 1 - Generating image for doc {doc_id}")
        eval_logger.debug(f"Generation prompt: {generation_prompt}")

        try:
            generator = torch.Generator(device="cuda").manual_seed(self.seed)

            image = self.sd_pipeline(
                prompt=generation_prompt,
                height=self.stage1_height,
                width=self.stage1_width,
                num_inference_steps=self.stage1_num_inference_steps,
                guidance_scale=self.stage1_guidance_scale,
                generator=generator,
            ).images[0]

            # Save image
            artifact_dir = os.path.join(self.intermediate_dir, task)
            os.makedirs(artifact_dir, exist_ok=True)
            image_path = os.path.join(
                artifact_dir, f"{doc_id}_stage1_generated.png"
            )
            image.save(image_path)

            eval_logger.debug(f"Stage 1 - Generated image saved to {image_path}")
            return [image_path]

        except Exception as e:
            eval_logger.error(f"Stage 1 failed for doc {doc_id}: {e}")
            if self.fail_gracefully:
                return []
            else:
                raise

    def _stage2_answer_with_image(
        self,
        question: str,
        image_path: str,
        doc_id: str,
        original_image: Optional[Image.Image] = None,
    ) -> str:
        """
        Stage 2: Answer question using generated image (and optionally original image)

        Args:
            question: Original question text
            image_path: Path to generated auxiliary image
            doc_id: Document ID for logging
            original_image: Original image (optional, used as primary reference)

        Returns:
            Answer text
        """
        eval_logger.debug(f"Stage 2 - Answering question for doc {doc_id}")
        eval_logger.debug(f"Question: {question}")

        try:
            # Load generated auxiliary image
            auxiliary_image = Image.open(image_path).convert("RGB")

            # Prepare messages with both images if original is provided
            if original_image is not None:
                eval_logger.debug(
                    "Stage 2 - Using both original and auxiliary images"
                )
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": original_image},
                            {"type": "image", "image": auxiliary_image},
                            {"type": "text", "text": question},
                        ],
                    }
                ]
            else:
                # Fallback to single image (auxiliary only)
                eval_logger.debug("Stage 2 - Using auxiliary image only")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": auxiliary_image},
                            {"type": "text", "text": question},
                        ],
                    }
                ]

            # Apply chat template
            text = self.qwen_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process inputs
            images = (
                [original_image, auxiliary_image]
                if original_image is not None
                else [auxiliary_image]
            )
            inputs = self.qwen_processor(
                text=[text], images=images, padding=True, return_tensors="pt"
            ).to("cuda")

            # Generate
            generated_ids = self.qwen_model_instance.generate(
                **inputs,
                max_new_tokens=self.stage2_max_new_tokens,
                temperature=self.stage2_temperature,
                do_sample=self.stage2_do_sample,
                top_p=self.stage2_top_p,
            )

            # Decode output
            generated_text = self.qwen_processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            # Remove the input prompt from output if present
            if text in generated_text:
                generated_text = generated_text.replace(text, "").strip()

            eval_logger.debug(
                f"Stage 2 - Generated answer: {generated_text[:100]}..."
            )
            return generated_text

        except Exception as e:
            eval_logger.error(f"Stage 2 failed for doc {doc_id}: {e}")
            if self.fail_gracefully:
                return ""
            else:
                raise

    def _save_intermediate_artifacts(
        self,
        doc_id: str,
        task: str,
        generation_prompt: str,
        generated_images: List[str],
        question: str,
        stage2_answer: str,
    ) -> None:
        """Save intermediate artifacts for debugging"""
        if not self.save_intermediate:
            return

        artifact_dir = os.path.join(self.intermediate_dir, task)
        os.makedirs(artifact_dir, exist_ok=True)

        # Save metadata
        metadata = {
            "doc_id": doc_id,
            "task": task,
            "generation_prompt": generation_prompt,
            "generated_images": generated_images,
            "question": question,
            "stage2_answer": stage2_answer,
        }

        metadata_path = os.path.join(artifact_dir, f"{doc_id}_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        eval_logger.debug(f"Saved intermediate artifacts to: {metadata_path}")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Main inference method implementing two-stage visual CoT

        Stage 1: Generate visualization image from text prompt
        Stage 2: Answer question using the generated image
        """
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="UniPic2VisualCoT Generating",
        )

        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = (
                request.args
            )

            # Extract original image from document using task_dict
            original_image = None
            if doc_to_visual is not None:
                try:
                    # Get doc from task_dict
                    doc = self.task_dict[task][split][doc_id]
                    original_visuals = doc_to_visual(doc)
                    if original_visuals and len(original_visuals) > 0:
                        original_image = original_visuals[0]
                        eval_logger.debug(
                            f"Extracted original image for doc {doc_id}"
                        )
                except Exception as e:
                    eval_logger.warning(
                        f"Failed to extract original image for doc {doc_id}: {e}"
                    )

            # Parse contexts to extract generation_prompt if provided
            gen_prompt_match = re.search(
                r"\[GEN_PROMPT\](.*?)\[/GEN_PROMPT\]", contexts, re.DOTALL
            )
            question_match = re.search(
                r"\[QUESTION\](.*?)\[/QUESTION\]", contexts, re.DOTALL
            )

            if gen_prompt_match and question_match:
                # Use custom generation prompt from task config
                custom_gen_prompt = gen_prompt_match.group(1).strip()
                actual_question = question_match.group(1).strip()
                generation_prompt = custom_gen_prompt.replace(
                    "{question}", actual_question
                )
                # Extract just the question for stage 2
                contexts = actual_question
                eval_logger.info("Using custom generation prompt from task config")
            else:
                # Use default template
                generation_prompt = self.generation_prompt_template.format(
                    question=contexts
                )

            eval_logger.info(f"\n{'='*60}")
            eval_logger.info(f"Processing doc {doc_id} from task {task}")
            eval_logger.info(f"{'='*60}")

            # Stage 1: Generate visualization image
            generated_images = self._stage1_generate_image(
                generation_prompt=generation_prompt, doc_id=doc_id, task=task
            )

            # Check if image was generated
            if not generated_images or len(generated_images) == 0:
                eval_logger.warning(
                    f"No image generated for doc {doc_id}, returning empty answer"
                )
                res.append("")
                pbar.update(1)
                continue

            # Stage 2: Answer question using generated image (and original image if available)
            final_answer = self._stage2_answer_with_image(
                question=contexts,
                image_path=generated_images[0],
                doc_id=doc_id,
                original_image=original_image,
            )

            # Save intermediate artifacts if enabled
            self._save_intermediate_artifacts(
                doc_id=doc_id,
                task=task,
                generation_prompt=generation_prompt,
                generated_images=generated_images,
                question=contexts,
                stage2_answer=final_answer,
            )

            # Return only final answer text
            res.append(final_answer)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError(
            "UniPic2VisualCoT is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue not yet implemented"""
        raise NotImplementedError(
            "Multi-round dialogue not yet implemented for UniPic2VisualCoT"
        )
