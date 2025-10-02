import tqdm
import clip
import torch.optim
from PIL import Image
from typing import Tuple
from transformers import logging
from utils.tensor import normalize
from utils.clip_embeddings import *
from utils.hpsv2 import initialize_hpsv2
from utils.vision import get_preprocessers
from torchvision.transforms.functional import to_pil_image
from utils.aesthetic import AestheticScorePredictor
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

logging.set_verbosity_error()

SDv1_5_PATH = "stable-diffusion-v1-5/stable-diffusion-v1-5"

def optimize_prompt(text_input_ids,
                    prompt_embeds_init,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    style_tensor,
                    style_scale,
                    pipe, latents, device,
                    number_of_iterations=10,
                    optimizer_type="Adam",
                    loss_term_coeffs=None,
                    optimizer_lr=0.01,
                    guidance_scale=7.5,
                    num_inference_steps=14,
                    model_name="stable-diffusion-v1-5/stable-diffusion-v1-5"
                    ) -> Tuple[Image.Image, Image.Image, list, float]:

    """
    Optimizes the given embedding
    :param text_input_ids: text tokens
    :param prompt_embeds_init: initial text embedding
    :param prompt_embeds: prompt_embeds embedding being optimized
    :param pooled_prompt_embeds: text embedding of the second text encoder (applicable in the case of sdxl and sdxl turbo)
    :param style_tensor: style image tensor (not used in the main optimization procedure)
    :param style_scale: style scale (not used in the main optimization procedure)
    :param pipe: diffusers diffusion pipeline
    :param latents: latents used for image generaiton
    :param device: device 
    :param number_of_iterations: number of iterations to run the prompt optimization
    :param optimizer_type: the name of the optimizer (e.g. Adam),
    :param loss_term_coeffs: coefficients for the optimization objective terms
    :param optimizer_lr: learning rate of the optimizer (e.g. 0.01),
    :param  guidance_scale: guidance scale used for CFG,
    :param num_inference_steps: number of inference steps to run the diffusion process,
    :param model_name: backbone T2I DM
    :return:
        Tuple[Image.Image, Image.Image, list, float]: A tuple containing:
        - The image generated with the optimized text embedding with lowest objective function value
        - The image generated with the optimized text embedding with the last optimiztion step
        - A list of objective function values 
        - The step with the best objetive function value
    """

    # max_for_pooler is assumed to be the same for both init embedding and optimized embedding
    max_for_pooler = text_input_ids.to(torch.int).argmax(dim=-1)
    bs = prompt_embeds_init.shape[0]

    if loss_term_coeffs is None:
        loss_term_coeffs = {
            "aes": 1.0,
            "text_image_sim": 1.0,
            "text_sim": 1.0,
            "hpsv2": 0.0,
        }
    print("loss_term_coeffs", loss_term_coeffs)

    if optimizer_type == "Adam":
        if pooled_prompt_embeds is not None:
            optimizer = torch.optim.Adam([prompt_embeds, pooled_prompt_embeds], lr=optimizer_lr)
        else:
            optimizer = torch.optim.Adam([prompt_embeds], lr=optimizer_lr)
    elif optimizer_type == "AdamW":
        if pooled_prompt_embeds is not None:
            optimizer = torch.optim.AdamW([prompt_embeds, pooled_prompt_embeds], lr=optimizer_lr)
        else:
            optimizer = torch.optim.AdamW([prompt_embeds], lr=optimizer_lr)
    elif optimizer_type == "SGD":
        if pooled_prompt_embeds is not None:
            optimizer = torch.optim.SGD([prompt_embeds, pooled_prompt_embeds], lr=optimizer_lr)
        else:
            optimizer = torch.optim.SGD([prompt_embeds], lr=optimizer_lr)
    elif optimizer_type == "LBFGS":
        if pooled_prompt_embeds is not None:
            optimizer = torch.optim.LBFGS([prompt_embeds, pooled_prompt_embeds],
                                          max_iter=5, lr=optimizer_lr)
        else:
            optimizer = torch.optim.LBFGS([prompt_embeds], max_iter=5, lr=optimizer_lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    if style_tensor is not None:
        style_tensor = style_tensor.to(device)
        assert style_scale is not None

    running_idx = 0

    if loss_term_coeffs["aes"] != 0.0 or loss_term_coeffs["text_image_sim"] != 0.0 or loss_term_coeffs["hpsv2"] != 0.0:
        clip_model, _ = clip.load("ViT-L/14", device=device)
        preprocess = get_preprocessers(224)
        aesthetic_score_predictor = AestheticScorePredictor(768)
        aesthetic_score_predictor.load_state_dict(torch.load("sac+logos+ava1-l14-linearMSE.pth"))
        aesthetic_score_predictor.to(device)
        aesthetic_score_predictor.eval()

        if pooled_prompt_embeds is None:
            vision_model_h14 = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            vision_transforms_h14 = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            vision_model_h14.to(device)
            vision_model_h14.eval()

    if loss_term_coeffs["hpsv2"] != 0.0:
        hpsv2, _ = initialize_hpsv2(device)
        hpsv2.eval()
    torch.autograd.set_detect_anomaly(True)

    best_loss = torch.tensor([[1000.0]], device=device, requires_grad=False)
    best_step = 0
    best_text_embeddings = None
    if pooled_prompt_embeds is not None:
        best_pooled_prompt_embeds = None

    def closure():
        optimizer.zero_grad()

        if pooled_prompt_embeds is not None:
            generated_tensor = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=None,
                                    pooled_prompt_embeds=pooled_prompt_embeds,
                                    negative_pooled_prompt_embeds=None,
                                    width=512, height=512,
                                    latents=latents, num_inference_steps=num_inference_steps,
                                    guidance_scale=guidance_scale)
        else:
            negative_prompt_embeds, = get_unconditional_embeddings_for_cfg([pipe.tokenizer],
                                                                           [pipe.text_encoder],
                                                                           device, 1,
                                                                           text_input_ids.shape[-1], False,
                                                                           prompt_embeds, None,
                                                                           1, False,
                                                                           model_name)

            generated_tensor = pipe(prompt_embeds=prompt_embeds,
                                    negative_prompt_embeds=negative_prompt_embeds,
                                    width=512, height=512,
                                    guidance_scale=guidance_scale,
                                    num_inference_steps=num_inference_steps,
                                    latents=latents)
        loss = torch.tensor([[0.0]], device=device, requires_grad=True)

        if loss_term_coeffs["aes"] != 0.0 or loss_term_coeffs["text_image_sim"] != 0.0:
            preprocessed_tensor = preprocess(generated_tensor)
            image_features_l14_normalized = normalize(
                clip_model.encode_image(preprocessed_tensor).type(torch.cuda.FloatTensor))

            if style_tensor is not None:
                style_features_l14_normalized = normalize(
                    clip_model.encode_image(preprocess(style_tensor)).type(torch.cuda.FloatTensor))

                image_features_l14_normalized = (1.0 - style_scale) * image_features_l14_normalized + style_scale * style_features_l14_normalized

                del style_features_l14_normalized

        if loss_term_coeffs["aes"] != 0.0:
            aes_score = 0.1 * aesthetic_score_predictor(image_features_l14_normalized)
            loss = loss - loss_term_coeffs["aes"] * aes_score

        if loss_term_coeffs["text_image_sim"] != 0.0 or loss_term_coeffs["text_sim"] != 0.0 or loss_term_coeffs[
            "hpsv2"] != 0.0:
            if pooled_prompt_embeds is not None:
                text_embeddings_normalized_for_img = prompt_embeds[:, :, :768][torch.arange(bs), max_for_pooler] @ clip_model.text_projection.type(torch.cuda.FloatTensor)
                text_embeddings_normalized_for_img = normalize(text_embeddings_normalized_for_img)
                text_embeddings_normalized = normalize(prompt_embeds[:, :, :768][torch.arange(bs), max_for_pooler])
            else:
                text_embeddings_normalized_for_img = prompt_embeds[torch.arange(bs), max_for_pooler] @ clip_model.text_projection.type(torch.cuda.FloatTensor)
                text_embeddings_normalized_for_img = normalize(text_embeddings_normalized_for_img)
                text_embeddings_normalized = normalize(prompt_embeds[torch.arange(bs), max_for_pooler])

        if loss_term_coeffs["hpsv2"] != 0.0:
            # not tested for SDXL-Turbo and v1.5
            preprocessed_hpsv2_tensor = preprocess(generated_tensor)
            hpsv2_outputs = hpsv2(preprocessed_hpsv2_tensor, text_input_ids.to(device))
            hps_score = torch.mm(hpsv2_outputs["image_features"],
                                 hpsv2_outputs["text_features"].T
                                 )
            loss = loss - loss_term_coeffs["hpsv2"] * hps_score

        if loss_term_coeffs["text_image_sim"] != 0.0:
            if pooled_prompt_embeds is not None or model_name == SDv1_5_PATH:
                sim = torch.mm(
                    image_features_l14_normalized,
                    text_embeddings_normalized_for_img.T
                )
            else:
                image_features_h14_normalized = normalize(vision_model_h14(
                    vision_transforms_h14(generated_tensor[0], return_tensors="pt")["pixel_values"].type(
                        torch.cuda.FloatTensor))["image_embeds"])

                if style_tensor is not None:
                    style_features_h14_normalized = normalize(
                        vision_model_h14(
                            vision_transforms_h14(style_tensor[None], return_tensors="pt")["pixel_values"].type(
                                torch.cuda.FloatTensor
                            )
                        )["image_embeds"]
                    )

                    image_features_h14_normalized = (1.0 - style_scale) * image_features_h14_normalized + style_scale * style_features_h14_normalized
                    del style_features_h14_normalized

                print("image_features_h14_normalized", image_features_h14_normalized.shape)
                print("text_embeddings_normalized", text_embeddings_normalized.shape)
                sim = torch.mm(
                    image_features_h14_normalized,
                    text_embeddings_normalized.T
                )

            loss = loss - loss_term_coeffs["text_image_sim"] * sim

        if loss_term_coeffs["text_sim"] != 0.0:
            text_score = torch.mm(
                normalize(
                    prompt_embeds_init[:, :, :text_embeddings_normalized.shape[1]][torch.arange(bs), max_for_pooler]),
                text_embeddings_normalized.T
            )
            loss = loss - loss_term_coeffs["text_sim"] * text_score

        nonlocal best_loss
        if loss < best_loss:
            best_loss = loss
            nonlocal best_step
            best_step = running_idx
            nonlocal best_text_embeddings
            best_text_embeddings = prompt_embeds.clone().detach()
            if pooled_prompt_embeds is not None:
                nonlocal best_pooled_prompt_embeds
                best_pooled_prompt_embeds = pooled_prompt_embeds.clone().detach()

        loss.backward()

        return loss

    progress_bar = tqdm.trange(number_of_iterations)
    losses = []
    for _ in progress_bar:
        loss_ = optimizer.step(closure)
        losses.append(loss_.item())

        progress_bar.set_description(f"loss = {loss_.item():.3}")
        running_idx += 1

    negative_prompt_embeds, = get_unconditional_embeddings_for_cfg([pipe.tokenizer],
                                                                   [pipe.text_encoder],
                                                                   device, 1,
                                                                   text_input_ids.shape[-1], False,
                                                                   prompt_embeds, None,
                                                                   1, False,
                                                                   model_name)

    if best_text_embeddings is not None:
        if pooled_prompt_embeds is not None:
            best_generated = pipe(prompt_embeds=best_text_embeddings,
                                  negative_prompt_embeds=None,
                                  pooled_prompt_embeds=best_pooled_prompt_embeds,
                                  negative_pooled_prompt_embeds=None,
                                  width=512, height=512,
                                  latents=latents, num_inference_steps=num_inference_steps,
                                  guidance_scale=guidance_scale)[0]
            best_generated = to_pil_image(best_generated)
        else:
            best_generated = pipe(prompt_embeds=best_text_embeddings,
                                  negative_prompt_embeds=negative_prompt_embeds,
                                  width=512, height=512,
                                  guidance_scale=guidance_scale,
                                  num_inference_steps=num_inference_steps,
                                  latents=latents)[0]
            best_generated = to_pil_image(best_generated)

        if pooled_prompt_embeds is not None:
            last_generated = pipe(prompt_embeds=prompt_embeds,
                                  negative_prompt_embeds=None,
                                  pooled_prompt_embeds=pooled_prompt_embeds,
                                  negative_pooled_prompt_embeds=None,
                                  width=512, height=512,
                                  latents=latents, num_inference_steps=num_inference_steps,
                                  guidance_scale=guidance_scale)[0]
            last_generated = to_pil_image(last_generated)
        else:
            last_generated = pipe(prompt_embeds=prompt_embeds,
                                  negative_prompt_embeds=negative_prompt_embeds,
                                  width=512, height=512,
                                  guidance_scale=guidance_scale,
                                  num_inference_steps=num_inference_steps,
                                  latents=latents)[0]
            last_generated = to_pil_image(last_generated)
    else:
        best_generated = None
        last_generated = None

    print("Optimization finished!")
    print(f"losses:", losses)

    return best_generated, last_generated, losses, best_step
