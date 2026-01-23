"""
Janus-Pro Visual Chain-of-Thought Model

Two-stage inference:
1. Stage 1: Generate visualization image from text prompt using Janus-Pro generation
2. Stage 2: Answer question using both original and generated images

Usage:
    python -m lmms_eval \
        --model janus_pro_visual_cot \
        --model_args pretrained=../models/Janus-Pro-7B \
        --tasks mme \
        --batch_size 1 \
        --device cuda:0
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Add Janus repository to Python path
wd = Path(__file__).parent.parent.parent.parent.resolve()
janus_path = os.path.join(str(wd), "Janus")
if os.path.exists(janus_path):
    sys.path.insert(0, janus_path)
    eval_logger.info(f"Added Janus path to sys.path: {janus_path}")


@register_model("janus_pro_visual_cot")
class JanusProVisualCoT(lmms):
    """
    Janus-Pro Visual Chain-of-Thought Model

    Performs two-stage visual reasoning:
    1. Generate visualization image from text prompt
    2. Answer question using both original and generated images
    """

    def __init__(
        self,
        pretrained: str = "../models/Janus-Pro-7B",
        device: str = "cuda",
        dtype: Optional[str] = "bfloat16",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        # Stage 1: Image generation parameters
        stage1_max_new_tokens: int = 16384,
        stage1_temperature: float = 1.0,
        stage1_cfg_weight: float = 5.0,
        stage1_image_token_num_per_image: int = 576,
        stage1_img_size: int = 384,
        stage1_patch_size: int = 16,
        # Stage 2: Visual understanding parameters
        stage2_max_new_tokens: int = 16384,
        stage2_temperature: float = 0.0,
        # Generation prompt template
        generation_prompt_template: str = "Generate a detailed visual diagram to help answer: {question}",
        # Output and debugging
        output_dir: Optional[str] = None,
        save_intermediate: bool = False,
        intermediate_dir: Optional[str] = None,
        fail_gracefully: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.pretrained = pretrained
        self.use_cache = use_cache
        self.trust_remote_code = trust_remote_code
        self.save_intermediate = save_intermediate
        self.fail_gracefully = fail_gracefully
        self.generation_prompt_template = generation_prompt_template

        # Stage 1 parameters
        self.stage1_max_new_tokens = stage1_max_new_tokens
        self.stage1_temperature = stage1_temperature
        self.stage1_cfg_weight = stage1_cfg_weight
        self.stage1_image_token_num_per_image = stage1_image_token_num_per_image
        self.stage1_img_size = stage1_img_size
        self.stage1_patch_size = stage1_patch_size

        # Stage 2 parameters
        self.stage2_max_new_tokens = stage2_max_new_tokens
        self.stage2_temperature = stage2_temperature

        # Setup output directories
        if output_dir is None:
            self.output_dir = "./logs/janus_pro_visual_cot"
        else:
            self.output_dir = output_dir

        self.generated_images_dir = os.path.join(self.output_dir, "generated_images")
        os.makedirs(self.generated_images_dir, exist_ok=True)

        if intermediate_dir is None:
            self.intermediate_dir = os.path.join(
                self.output_dir, "intermediate_artifacts"
            )
        else:
            self.intermediate_dir = intermediate_dir

        if save_intermediate:
            os.makedirs(self.intermediate_dir, exist_ok=True)
            eval_logger.info(
                f"Intermediate artifacts will be saved to: {self.intermediate_dir}"
            )

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
            self._dtype = torch.bfloat16

        # Load model
        eval_logger.info(f"Loading Janus-Pro model from {pretrained}")
        self._load_model(pretrained, attn_implementation)

        self.batch_size_per_gpu = int(batch_size)
        assert self.batch_size_per_gpu == 1, "batch_size > 1 not supported"

        # Setup distributed training
        if accelerator.num_processes > 1:
            distributed_type_list = [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ]
            assert accelerator.distributed_type in distributed_type_list
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
            eval_logger.info(f"Using single device: {self._device}")
            self._rank = 0
            self._world_size = 1

        eval_logger.info("Janus-Pro Visual CoT model initialized successfully")

    def _load_model(self, pretrained: str, attn_implementation: Optional[str]):
        """Load Janus-Pro model and processor."""
        try:
            from transformers import AutoModelForCausalLM
            from janus.models import MultiModalityCausalLM, VLChatProcessor

            eval_logger.info("Using Janus library for model loading")

            # Load processor
            self._processor = VLChatProcessor.from_pretrained(pretrained)
            self._tokenizer = self._processor.tokenizer

            # Load model
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

            self._model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                trust_remote_code=True,
                torch_dtype=self._dtype,
            )
            self._model = self._model.to(self._device).eval()
            self._config = self._model.config

            eval_logger.info("Janus-Pro model loaded successfully")

        except ImportError as e:
            raise ImportError(
                f"Failed to import Janus library. Please install it:\n"
                f"  cd lmms-eval && git clone https://github.com/deepseek-ai/Janus.git\n"
                f"Error: {e}"
            )

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

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

    def flatten(self, input_list):
        new_list = []
        for i in input_list:
            for j in i:
                new_list.append(j)
        return new_list

    def _extract_image_from_various_formats(self, img_data) -> Optional[Image.Image]:
        """Extract PIL Image from various formats."""
        try:
            if img_data is None:
                return None
            elif isinstance(img_data, Image.Image):
                return img_data.convert("RGB")
            elif isinstance(img_data, str):
                return Image.open(img_data).convert("RGB")
            elif isinstance(img_data, dict):
                if "bytes" in img_data:
                    from io import BytesIO

                    return Image.open(BytesIO(img_data["bytes"])).convert("RGB")
                elif "path" in img_data:
                    return Image.open(img_data["path"]).convert("RGB")
                elif "image" in img_data:
                    return self._extract_image_from_various_formats(img_data["image"])
            else:
                return Image.open(img_data).convert("RGB")
        except Exception as e:
            eval_logger.debug(f"Failed to extract image: {e}")
            return None

    def _stage1_generate_image(
        self, generation_prompt: str, doc_id: str, task: str, original_image=None
    ) -> Tuple[str, List[str]]:
        """Stage 1: Generate visualization image using Janus-Pro generation."""
        try:
            original_image = self._extract_image_from_various_formats(original_image)

            # Build conversation for generation
            images = [original_image] if original_image else []
            if images:
                image_placeholders = "<image_placeholder>\n" * len(images)
                user_content = image_placeholders + generation_prompt
            else:
                user_content = generation_prompt

            conversation = [
                {
                    "role": "User",
                    "content": user_content,
                    "images": images,
                },
                {"role": "Assistant", "content": ""},
            ]

            # Apply SFT template and add image_start_tag to trigger generation
            sft_format = self._processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self._processor.sft_format,
                system_prompt="",
            )
            prompt = sft_format + self._processor.image_start_tag

            # Tokenize prompt
            input_ids = self._tokenizer.encode(prompt)
            input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self._device)

            # Generate image tokens using CFG
            with torch.no_grad():
                generated_tokens = self._generate_image_tokens_with_cfg(input_ids)

                if generated_tokens.shape[1] >= self.stage1_image_token_num_per_image:
                    # Decode image from tokens
                    gen_img_tensor = generated_tokens[
                        :, : self.stage1_image_token_num_per_image
                    ]

                    generated_image = self._model.gen_vision_model.decode_code(
                        gen_img_tensor.to(dtype=torch.int),
                        shape=[
                            1,
                            8,
                            self.stage1_img_size // self.stage1_patch_size,
                            self.stage1_img_size // self.stage1_patch_size,
                        ],
                    )

                    # Convert to PIL Image
                    generated_image = (
                        generated_image[0].permute(1, 2, 0).cpu().float().numpy()
                    )
                    generated_image = (
                        ((generated_image + 1) / 2 * 255).clip(0, 255).astype("uint8")
                    )
                    generated_image = Image.fromarray(generated_image)

                    # Save generated image
                    task_dir = os.path.join(self.generated_images_dir, task)
                    os.makedirs(task_dir, exist_ok=True)
                    image_path = os.path.join(task_dir, f"{doc_id}_gen.png")
                    generated_image.save(image_path)

                    eval_logger.info(f"Generated image saved to {image_path}")

                    del generated_tokens, gen_img_tensor
                    torch.cuda.empty_cache()

                    return prompt, [image_path]
                else:
                    eval_logger.warning(
                        f"Not enough image tokens generated: {generated_tokens.shape[1]}"
                    )
                    del generated_tokens
                    torch.cuda.empty_cache()
                    return prompt, []

        except Exception as e:
            eval_logger.error(f"Stage 1 generation error: {e}")
            import traceback

            eval_logger.error(traceback.format_exc())
            if self.fail_gracefully:
                return "", []
            raise

    def _generate_image_tokens_with_cfg(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate image tokens using classifier-free guidance with KV cache optimization."""
        # Create conditional and unconditional inputs
        batch_size = input_ids.shape[0]
        tokens = torch.zeros((batch_size * 2, input_ids.shape[1]), dtype=torch.long).to(
            self._device
        )
        tokens[0::2] = input_ids  # conditional
        tokens[1::2] = input_ids.clone()
        tokens[1::2, 1:-1] = self._processor.pad_id  # unconditional (mask content)

        inputs_embeds = self._model.language_model.get_input_embeddings()(tokens)

        generated_tokens = torch.zeros(
            (batch_size, self.stage1_image_token_num_per_image), dtype=torch.long
        ).to(self._device)

        # Initialize KV cache
        past_key_values = None

        for i in range(self.stage1_image_token_num_per_image):
            # Use KV cache for faster generation
            if past_key_values is None:
                # First iteration: process full sequence
                model_outputs = self._model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                )
            else:
                # Subsequent iterations: only process last token
                model_outputs = self._model.language_model.model(
                    inputs_embeds=inputs_embeds[:, -1:, :],
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            hidden_states = model_outputs.last_hidden_state
            past_key_values = model_outputs.past_key_values

            logits = self._model.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]

            # Apply CFG
            logits = logit_uncond + self.stage1_cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / self.stage1_temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            # Prepare next input
            next_token_doubled = torch.cat([next_token, next_token], dim=0)
            img_embeds = self._model.prepare_gen_img_embeds(next_token_doubled)
            # Ensure img_embeds has shape [batch*2, 1, hidden_dim]
            if img_embeds.dim() == 2:
                img_embeds = img_embeds.unsqueeze(1)
            elif img_embeds.dim() == 4:
                img_embeds = img_embeds.squeeze(1)
            inputs_embeds = img_embeds

        return generated_tokens

    def _stage2_answer_with_images(
        self, question: str, generated_image_path: str, original_image=None
    ) -> str:
        """Stage 2: Answer question using both original and generated images."""
        try:
            # Load images
            images = []
            if original_image:
                original_image = self._extract_image_from_various_formats(
                    original_image
                )
                if original_image:
                    images.append(original_image)

            gen_image = Image.open(generated_image_path).convert("RGB")
            images.append(gen_image)

            # Build conversation
            image_placeholders = "<image_placeholder>\n" * len(images)
            user_content = image_placeholders + question

            conversation = [
                {
                    "role": "User",
                    "content": user_content,
                    "images": images,
                },
                {"role": "Assistant", "content": ""},
            ]

            prepare_inputs = self._processor(
                conversations=conversation,
                images=images,
                force_batchify=True,
            ).to(self._device)

            inputs_embeds = self._model.prepare_inputs_embeds(**prepare_inputs)

            # Generate answer
            with torch.no_grad():
                outputs = self._model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self._tokenizer.eos_token_id,
                    bos_token_id=self._tokenizer.bos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    max_new_tokens=self.stage2_max_new_tokens,
                    do_sample=self.stage2_temperature > 0,
                    temperature=self.stage2_temperature
                    if self.stage2_temperature > 0
                    else None,
                    use_cache=self.use_cache,
                )

            answer = self._tokenizer.decode(
                outputs[0].cpu().tolist(), skip_special_tokens=True
            )

            del outputs, inputs_embeds, prepare_inputs
            torch.cuda.empty_cache()

            return answer

        except Exception as e:
            eval_logger.error(f"Stage 2 error: {e}")
            import traceback

            eval_logger.error(traceback.format_exc())
            if self.fail_gracefully:
                return ""
            raise

    def _save_intermediate_artifacts(
        self,
        doc_id: str,
        task: str,
        generation_prompt: str,
        stage1_text: str,
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
            "stage1_text": stage1_text,
            "generated_images": generated_images,
            "question": question,
            "stage2_answer": stage2_answer,
        }

        metadata_path = os.path.join(artifact_dir, f"{doc_id}_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        eval_logger.debug(f"Saved intermediate artifacts to: {metadata_path}")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate text until stopping criteria are met using two-stage visual CoT."""
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Janus-Pro Visual CoT",
        )

        re_ords = utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            (
                contexts,
                all_gen_kwargs,
                doc_to_visual,
                doc_id,
                task,
                split,
            ) = zip(*chunk)
            task = task[0]
            split = split[0]
            doc_id = doc_id[0]
            contexts = contexts[0]

            # Extract original image
            original_image = None
            if doc_to_visual[0]:
                visuals = doc_to_visual[0](self.task_dict[task][split][doc_id])
                if visuals:
                    original_image = visuals[0]

            # Stage 1: Generate visualization image
            generation_prompt = self.generation_prompt_template.format(
                question=contexts
            )
            stage1_text, generated_images = self._stage1_generate_image(
                generation_prompt, doc_id, task, original_image
            )

            # Stage 2: Answer with both images
            if generated_images:
                final_answer = self._stage2_answer_with_images(
                    contexts, generated_images[0], original_image
                )
            else:
                eval_logger.warning(
                    f"No image generated for {doc_id}, skipping stage 2"
                )
                final_answer = ""

            # Save intermediate artifacts if enabled
            self._save_intermediate_artifacts(
                doc_id=doc_id,
                task=task,
                generation_prompt=generation_prompt,
                stage1_text=stage1_text,
                generated_images=generated_images,
                question=contexts,
                stage2_answer=final_answer,
            )

            res.append(final_answer)
            self.cache_hook.add_partial(
                "generate_until", (contexts, all_gen_kwargs[0]), final_answer
            )
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_uni_mmmu_interleaved(
        self,
        input_images: List,
        prompt: str,
        doc_id: str,
        task: str,
        interleaved_config: dict,
        doc: dict = None,
    ) -> Tuple[str, List[str]]:
        """
        Uni-MMMU interleaved generation for Janus-Pro Visual CoT.

        This implements the exact generation flow from the original Uni-MMMU:
        - Jigsaw: gen_image(cand0) → gen_image(cand1) → gen_text(answer)
        - Maze/Sliding: [gen_text(plan) → gen_image(step)]×k → gen_text(answer)

        Args:
            input_images: List of input images
            prompt: Base prompt text
            doc_id: Document ID for file naming
            task: Task name for file naming
            interleaved_config: Configuration dict from yaml
            doc: Document data for dynamic num_images extraction

        Returns:
            Tuple of (final_text_answer, list_of_generated_image_paths)
        """
        import json as json_module

        task_type = interleaved_config.get("task_type", "jigsaw")

        # Get num_images dynamically from doc if available
        num_images = interleaved_config.get("num_images", 2)
        if doc is not None:
            if task_type == "maze":
                # Get step count from ground truth
                steps_str = doc.get("steps", "[]")
                steps = json_module.loads(steps_str) if isinstance(steps_str, str) else steps_str
                if steps:
                    num_images = len(steps)
            elif task_type == "sliding":
                # Get step count from ground truth
                steps_str = doc.get("steps_words", "[]")
                steps = json_module.loads(steps_str) if isinstance(steps_str, str) else steps_str
                if steps:
                    num_images = len(steps)

        # Extract original image from input_images
        original_image = None
        if input_images and len(input_images) > 0:
            original_image = self._extract_image_from_various_formats(input_images[0])

        generated_images = []

        if task_type == "jigsaw":
            # Jigsaw: Generate 2 completed images then final answer
            # Image 1: Candidate 0 completion
            suffix1 = "Output ONLY a single image with Candidate 0 placed in the bottom-right cell. No text."
            gen_prompt1 = prompt + "\n\n" + suffix1

            _, img_paths_0 = self._stage1_generate_image(
                generation_prompt=gen_prompt1,
                doc_id=f"{doc_id}_cand0",
                task=task,
                original_image=original_image,
            )
            if img_paths_0:
                generated_images.extend(img_paths_0)
                eval_logger.info(f"Saved jigsaw image 0: {img_paths_0[0]}")

            # Image 2: Candidate 1 completion
            suffix2 = "Output ONLY a single image with Candidate 1 placed in the bottom-right cell. No text."
            gen_prompt2 = prompt + "\n\n" + suffix2

            _, img_paths_1 = self._stage1_generate_image(
                generation_prompt=gen_prompt2,
                doc_id=f"{doc_id}_cand1",
                task=task,
                original_image=original_image,
            )
            if img_paths_1:
                generated_images.extend(img_paths_1)
                eval_logger.info(f"Saved jigsaw image 1: {img_paths_1[0]}")

            # Final answer using stage 2 with all generated images
            final_suffix = (
                'Now output EXACTLY ONE <FINAL_ANSWER_JSON>{"choice": 0 or 1, "rationale": "≤30 words"}</FINAL_ANSWER_JSON>\n'
                "Do not output any additional images."
            )
            final_question = prompt + "\n\n" + final_suffix

            # Use stage 2 to answer with the generated images
            if len(generated_images) >= 2:
                # For jigsaw, we want to use both generated images
                # Build images list: original + both generated
                images = []
                if original_image:
                    images.append(original_image)

                # Load both generated images
                for img_path in generated_images[:2]:
                    gen_img = Image.open(img_path).convert("RGB")
                    images.append(gen_img)

                # Build conversation
                image_placeholders = "<image_placeholder>\n" * len(images)
                user_content = image_placeholders + final_question

                conversation = [
                    {
                        "role": "User",
                        "content": user_content,
                        "images": images,
                    },
                    {"role": "Assistant", "content": ""},
                ]

                prepare_inputs = self._processor(
                    conversations=conversation,
                    images=images,
                    force_batchify=True,
                ).to(self._device)

                inputs_embeds = self._model.prepare_inputs_embeds(**prepare_inputs)

                # Generate answer
                with torch.no_grad():
                    outputs = self._model.language_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=prepare_inputs.attention_mask,
                        pad_token_id=self._tokenizer.eos_token_id,
                        bos_token_id=self._tokenizer.bos_token_id,
                        eos_token_id=self._tokenizer.eos_token_id,
                        max_new_tokens=self.stage2_max_new_tokens,
                        do_sample=False,
                        use_cache=self.use_cache,
                    )

                final_text = self._tokenizer.decode(
                    outputs[0].cpu().tolist(), skip_special_tokens=True
                )

                del outputs, inputs_embeds, prepare_inputs
                torch.cuda.empty_cache()
            else:
                final_text = ""

        else:
            # Maze/Sliding: [gen_text(plan) → gen_image(step)]×k → gen_text(answer)
            for i in range(1, num_images + 1):
                # Generate step image with planning prompt
                if task_type == "maze":
                    plan_suffix = f'Step {i}: Generate an image showing the next move (one step up/down/left/right).'
                else:  # sliding
                    plan_suffix = f'Step {i}: Generate an image showing which tile to move and in which direction.'

                gen_prompt = prompt + "\n\n" + plan_suffix

                _, img_paths = self._stage1_generate_image(
                    generation_prompt=gen_prompt,
                    doc_id=f"{doc_id}_step_{i:04d}",
                    task=task,
                    original_image=original_image,
                )

                if img_paths:
                    generated_images.extend(img_paths)
                    eval_logger.info(f"Saved step {i} image: {img_paths[0]}")

            # Final answer using all generated step images
            final_suffix = (
                "After the images, emit EXACTLY ONE LINE containing ONLY the final move list "
                "as <ANSWER_JSON>[...]</ANSWER_JSON>. No other text."
            )
            final_question = prompt + "\n\n" + final_suffix

            # Use stage 2 to answer with all generated images
            if generated_images:
                # Build images list: original + all generated steps
                images = []
                if original_image:
                    images.append(original_image)

                # Load all generated images
                for img_path in generated_images:
                    gen_img = Image.open(img_path).convert("RGB")
                    images.append(gen_img)

                # Build conversation
                image_placeholders = "<image_placeholder>\n" * len(images)
                user_content = image_placeholders + final_question

                conversation = [
                    {
                        "role": "User",
                        "content": user_content,
                        "images": images,
                    },
                    {"role": "Assistant", "content": ""},
                ]

                prepare_inputs = self._processor(
                    conversations=conversation,
                    images=images,
                    force_batchify=True,
                ).to(self._device)

                inputs_embeds = self._model.prepare_inputs_embeds(**prepare_inputs)

                # Generate answer
                with torch.no_grad():
                    outputs = self._model.language_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=prepare_inputs.attention_mask,
                        pad_token_id=self._tokenizer.eos_token_id,
                        bos_token_id=self._tokenizer.bos_token_id,
                        eos_token_id=self._tokenizer.eos_token_id,
                        max_new_tokens=self.stage2_max_new_tokens,
                        do_sample=False,
                        use_cache=self.use_cache,
                    )

                final_text = self._tokenizer.decode(
                    outputs[0].cpu().tolist(), skip_special_tokens=True
                )

                del outputs, inputs_embeds, prepare_inputs
                torch.cuda.empty_cache()
            else:
                final_text = ""

        return final_text, generated_images

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood not implemented")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation not implemented")