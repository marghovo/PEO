import os
from utils.clip_embeddings import *
from argparse import ArgumentParser
from utils.filenames import clean_filename
from optimize import optimize_prompt
from diffusers import UniPCMultistepScheduler
from pipelines import POPipeline, POSDXLPipeline
from torchvision.transforms.functional import to_pil_image

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=555)
parser.add_argument("--optimizer_type", type=str, default="Adam")
parser.add_argument("--optimizer_lr", type=float, default=0.01)
parser.add_argument("--aes", type=float, default=1.0)
parser.add_argument("--text_image_sim", type=float, default=0.5)
parser.add_argument("--text_sim", type=float, default=0.5)
parser.add_argument("--hpsv2", type=float, default=0.0)
parser.add_argument("--guidance_scale", type=float, default=7.5)
parser.add_argument("--num_inference_steps", type=int, default=14)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--logdir", type=str, default="results")
parser.add_argument("--prompt_path", type=str, default="prompts.txt")
parser.add_argument("--model_name", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")

if __name__ == "__main__":
    args = parser.parse_args()
    seed = args.seed
    device = args.device
    dtype = torch.float32
    with open(args.prompt_path, 'r') as file:
        file_contents = file.read()
    prompts = file_contents.split("\n")

    os.makedirs(f"{args.logdir}", exist_ok=True)

    if "sdxl" in args.model_name:
        pipe = POSDXLPipeline.from_pretrained(args.model_name, torch_dtype=dtype).to(device)
    else:
        pipe = POPipeline.from_pretrained(args.model_name, torch_dtype=dtype).to(device)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.set_progress_bar_config(disable=True)
    pipe.enable_xformers_memory_efficient_attention()

    print("Number of prompts to optimize:", len(prompts))
    for prompt_idx, prompt in enumerate(prompts):
        generator = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn((1, 4, 64, 64), device=args.device, dtype=dtype, generator=generator).repeat(1, 1, 1, 1)
        latents_clone = latents.clone().detach().requires_grad_(True)

        pooled_prompt_embeds = None
        if "sdxl" in args.model_name:
            text_inputs = pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids

            text_inputs_2 = pipe.tokenizer_2(
                prompt,
                padding="max_length",
                max_length=pipe.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids_2 = text_inputs_2.input_ids

            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                [text_input_ids, text_input_ids_2],
                [pipe.text_encoder, pipe.text_encoder_2],
                "cuda",
                1,
                None,
                return_pooled=True,
                model_name="sdxl"
            )
            prompt_embeds_init = prompt_embeds.clone().detach().requires_grad_(True)
            prompt_embeds = prompt_embeds.clone().detach().requires_grad_(True)

            pooled_prompt_embeds = pooled_prompt_embeds.clone().detach().requires_grad_(True)

            generated_tensor = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=None,
                                    pooled_prompt_embeds=pooled_prompt_embeds,
                                    negative_pooled_prompt_embeds=None,
                                    width=512, height=512,
                                    latents=latents, num_inference_steps=args.num_inference_steps,
                                    guidance_scale=args.guidance_scale)[0]
        else:
            text_inputs = pipe.tokenizer(
                prompts[prompt_idx],
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds, = encode_prompt([text_input_ids],
                                           [pipe.text_encoder],
                                           pipe.device, 1,
                                           None, False, args.model_name)

            prompt_embeds_init = prompt_embeds.clone().detach().requires_grad_(True)
            prompt_embeds = prompt_embeds.clone().detach().requires_grad_(True)

            negative_prompt_embeds, = get_unconditional_embeddings_for_cfg([pipe.tokenizer],
                                                                           [pipe.text_encoder],
                                                                           device, 1,
                                                                           text_input_ids.shape[-1], False,
                                                                           prompt_embeds, None,
                                                                           1, False, args.model_name)
            generated_tensor = pipe(prompt_embeds=prompt_embeds,
                                    negative_prompt_embeds=negative_prompt_embeds,
                                    width=512, height=512,
                                    guidance_scale=args.guidance_scale,
                                    num_inference_steps=args.num_inference_steps,
                                    latents=latents)[0]

        saving_dir = f"{args.logdir}/{clean_filename(prompt)}_{seed}"

        to_pil_image(generated_tensor).save(f"{saving_dir}_{0}.png")

        del generated_tensor
        del latents

        if "sdxl" not in args.model_name:
            del negative_prompt_embeds

        loss_term_coeffs = {
            "aes": args.aes,
            "text_image_sim": args.text_image_sim,
            "text_sim": args.text_sim,
            "hpsv2": args.hpsv2,
        }

        optimized, _, _, _ = optimize_prompt(text_input_ids, prompt_embeds_init,
                                             prompt_embeds, pooled_prompt_embeds,
                                             None, None,
                                             pipe, latents_clone, device,
                                             optimizer_type=args.optimizer_type,
                                             loss_term_coeffs=loss_term_coeffs,
                                             optimizer_lr=args.optimizer_lr,
                                             guidance_scale=args.guidance_scale,
                                             num_inference_steps=args.num_inference_steps,
                                             model_name=args.model_name
                                             )

        optimized.save(f"{saving_dir}_optimized.png")

print("done")
