# heavly relies on https://github.com/huggingface/diffusers and https://github.com/tgxs002/HPSv2

import torch

def get_unconditional_embeddings_for_cfg(tokenizers, text_encoders, device,
                                         num_images_per_prompt,
                                         max_length, force_zeros_for_empty_prompt,
                                         prompt_embeds, pooled_prompt_embeds,
                                         batch_size=1, return_pooled=False, model_name="sdxl"):
    zero_out_negative_prompt = force_zeros_for_empty_prompt
    if zero_out_negative_prompt:
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        if return_pooled:
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
    else:
        negative_prompt = ""
        if "sdxl" in model_name:
            negative_prompt_2 = negative_prompt
            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )
            uncond_tokens = [negative_prompt, negative_prompt_2]
        else:
            uncond_tokens = [negative_prompt]

        negative_prompt_embeds_list = []
        for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
            uncond_input = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            negative_prompt_embeds = text_encoder(
                uncond_input.input_ids.to(device),
                output_hidden_states=True,
            )
            negative_pooled_prompt_embeds = negative_prompt_embeds[0]
            if "sdxl" in model_name:
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
            else:
                negative_prompt_embeds = negative_prompt_embeds[0]

            negative_prompt_embeds_list.append(negative_prompt_embeds)

        negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

    # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
    seq_len = negative_prompt_embeds.shape[1]
    # negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoders[0].dtype, device=device)
    negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
    negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    outputs = (negative_prompt_embeds,)
    if return_pooled:
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            prompt_embeds.shape[0] * num_images_per_prompt, -1
        )
        outputs = outputs + (negative_pooled_prompt_embeds,)

    return outputs


def encode_prompt(text_input_ids_all, text_encoders, device,
                  num_images_per_prompt, clip_skip, return_pooled=False,
                  model_name="sdxl"):
    prompt_embeds_list = []
    for text_input_ids, text_encoder in zip(text_input_ids_all, text_encoders):
        if "sdxl" in model_name:
            prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        else:
            prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)
        # We are only ALWAYS interested in the pooled output of the final text encoder
        if return_pooled:
            pooled_prompt_embeds = prompt_embeds[0]

        if "sdxl" in model_name:
            if clip_skip is None:
                prompt_embeds = prompt_embeds.hidden_states[-2]
            else:
                # "2" because SDXL always indexes from the penultimate layer.
                prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]
        else:
            prompt_embeds = prompt_embeds[0]

        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

    # prompt_embeds = prompt_embeds.to(dtype=text_encoders[0].dtype, device=device)
    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    outputs = (prompt_embeds,)
    if return_pooled:
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        outputs = outputs + (pooled_prompt_embeds,)

    return outputs