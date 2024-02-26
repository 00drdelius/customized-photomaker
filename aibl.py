import torch
import numpy as np
import random
import os
import sys

from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler

from huggingface_hub import hf_hub_download
# import spaces
import gradio as gr

from photomaker import PhotoMakerStableDiffusionXLPipeline
from style_template import styles

# global variable
base_model_path = 'models/models--SG161222--RealVisXL_V3.0/snapshots/11ee564ebf4bd96d90ed5d473cb8e7f2e6450bcf'
try:
    if torch.cuda.is_available():
        device = "cuda"
    elif sys.platform == "darwin" and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
except:
    device = "cpu"

MAX_SEED = np.iinfo(np.int32).max
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Photographic (Default)"

# download PhotoMaker checkpoint to cache
# photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")

# if device == "mps":
#     torch_dtype = torch.float16
# else:
#     torch_dtype = torch.bfloat16
# pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
#     base_model_path, 
#     torch_dtype=torch_dtype,
#     use_safetensors=True, 
#     variant="fp16",
#     # local_files_only=True,
# ).to(device)

# pipe.load_photomaker_adapter(
#     os.path.dirname(photomaker_ckpt),
#     subfolder="",
#     weight_name=os.path.basename(photomaker_ckpt),
#     trigger_word="img"
# )
# pipe.id_encoder.to(device)

# pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
# # pipe.set_adapters(["photomaker"], adapter_weights=[1.0])
# pipe.fuse_lora()

# @spaces.GPU(enable_queue=True)
# def generate_image(upload_images, prompt, negative_prompt, style_name, num_steps, style_strength_ratio, num_outputs, guidance_scale, seed, progress=gr.Progress(track_tqdm=True)):
#     # check the trigger word
#     image_token_id = pipe.tokenizer.convert_tokens_to_ids(pipe.trigger_word)
#     input_ids = pipe.tokenizer.encode(prompt)
#     if image_token_id not in input_ids:
#         raise gr.Error(f"Cannot find the trigger word '{pipe.trigger_word}' in text prompt! Please refer to step 2️⃣")

#     if input_ids.count(image_token_id) > 1:
#         raise gr.Error(f"Cannot use multiple trigger words '{pipe.trigger_word}' in text prompt!")

#     # apply the style template
#     prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

#     if upload_images is None:
#         raise gr.Error(f"Cannot find any input face image! Please refer to step 1️⃣")

#     input_id_images = []
#     for img in upload_images:
#         input_id_images.append(load_image(img))
    
#     generator = torch.Generator(device=device).manual_seed(seed)

#     print("Start inference...")
#     print(f"[调试] 提示词: {prompt}, \n[调试] 反向提示词: {negative_prompt}")
#     start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
#     if start_merge_step > 30:
#         start_merge_step = 30
#     print(start_merge_step)
#     images = pipe(
#         prompt=prompt,
#         input_id_images=input_id_images,
#         negative_prompt=negative_prompt,
#         num_images_per_prompt=num_outputs,
#         num_inference_steps=num_steps,
#         start_merge_step=start_merge_step,
#         generator=generator,
#         guidance_scale=guidance_scale,
#     ).images
#     return images, gr.update(visible=True)

def generate_image(upload_images, prompt, negative_prompt, style_name, num_steps, style_strength_ratio, num_outputs, guidance_scale, seed, progress=gr.Progress(track_tqdm=True)):
    images = open("examples/lenna_woman/lenna.jpg",'rb').read()
    return images, gr.update(visible=True)

def swap_to_gallery(images):
    return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=False)

def upload_example_to_gallery(images, prompt, style, negative_prompt):
    return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=False)

def remove_back_to_files():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
    
def remove_tips():
    return gr.update(visible=False)

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def apply_style(style_name: str, positive: str, negative: str = ""):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + ' ' + negative

def get_image_path_list(folder_name):
    image_basename_list = os.listdir(folder_name)
    image_path_list = sorted([os.path.join(folder_name, basename) for basename in image_basename_list])
    return image_path_list

def get_example():
    case = [
        [
            get_image_path_list('./examples/scarletthead_woman'),
            "instagram photo, portrait photo of a woman img, colorful, perfect face, natural skin, hard shadows, film grain",
            "(No style)",
            "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth",
        ],
        [
            get_image_path_list('./examples/newton_man'),
            "sci-fi, closeup portrait photo of a man img wearing the sunglasses in Iron man suit, face, slim body, high quality, film grain",
            "(No style)",
            "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth",
        ],
        [
            get_image_path_list('./examples/yangmi_woman'),
            "White dress, full body shot, portrait photo of a woman img, perfect face, natural skin",
            "(No style)",
            "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth",
        ],
        
    ]
    return case

### Description and style
logo = r"""
<center><img src='https://photo-maker.github.io/assets/logo.png' alt='PhotoMaker logo' style="width:80px; margin-bottom:10px"></center>
"""
title = r"""
<h1 align="center">PhotoMaker：通过上传照片生成逼真的人类照片</h1>
"""

description = r"""
❗️❗️❗️[<b>重要</b>] 步骤：<br>
1️⃣ 上传你想定制的人的图片。上传一张图片就可以，但多张更好。虽然我们不进行面部检测，但上传图片中的面部应该<b>占据图片的大部分</b>。<br>
2️⃣ 输入一个文本提示，确保遵循你想定制的<b>类别词</b>，并使用<b>触发词</b> img，比如：man img 或 woman img 或 girl img。<br>
3️⃣ 选择你喜欢的风格模板。<br>
4️⃣ 点击<b>提交</b>按钮开始定制。
"""

article = r"""
下载地址：<a href='https://www.aibl.vip' target='_blank'>AIBL论坛</a>
"""

tips = r"""
### PhotoMaker使用技巧
1. 上传更多要定制的人的照片以提高身份保真度。如果输入是亚洲面孔，可以考虑在类别词前添加 'asian' ，例如`asian woman img`
2. 在进行风格化时，生成的面孔是否看起来太逼真了？尝试切换到我们的另一个 gradio 演示 [PhotoMaker-Style](https://huggingface.co/spaces/TencentARC/PhotoMaker-Style). 调整风格强度到30-50，数值越大，身份保真度越低，但风格化能力会更好.
3. 为了更快的速度，减少生成图像的数量和采样步骤。但请注意，减少采样步骤可能会影响身份保真度.
"""
# We have provided some generate examples and comparisons at: [this website]().
# 3. Don't make the prompt too long, as we will trim it if it exceeds 77 tokens. 
# 4. When generating realistic photos, if it's not real enough, try switching to our other gradio application [PhotoMaker-Realistic]().

css = '''
.gradio-container {width: 85% !important}
'''
with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    # gr.DuplicateButton(
    #     value="Duplicate Space for private use ",
    #     elem_id="duplicate-button",
    #     visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    # )
    with gr.Row():
        with gr.Column():
            files = gr.Files(
                        label="拖动（选择）1张或多张您的面部照片",
                        file_types=["image"]
                    )
            uploaded_files = gr.Gallery(label="您的图片", visible=False, columns=5, rows=1, height=200)
            with gr.Column(visible=False) as clear_button:
                remove_and_reupload = gr.ClearButton(value="移除并上传新的图像", components=files, size="sm")
            style = gr.Dropdown(label="风格模板", choices=STYLE_NAMES, value=DEFAULT_STYLE_NAME)
            submit = gr.Button("提交")

            with gr.Accordion(open=False, label="高级选项"):
                prompt = gr.Textbox(
                    label="正向提示词",
                    info="尝试使用类似 'a photo of a man/woman img' 的表达，'img' 是触发词",
                    placeholder="A photo of a [man/woman img]..."
                )
                negative_prompt = gr.Textbox(
                    label="反向提示词", 
                    placeholder="low quality",
                    value="nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
                )
                num_steps = gr.Slider( 
                    label="采样步数",
                    minimum=20,
                    maximum=100,
                    step=1,
                    value=50,
                )
                style_strength_ratio = gr.Slider(
                    label="风格强度 (%)",
                    minimum=15,
                    maximum=50,
                    step=1,
                    value=20,
                )
                num_outputs = gr.Slider(
                    label="输出图像数量",
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=2,
                )
                guidance_scale = gr.Slider(
                    label="引导比例(Guidance scale)",
                    minimum=0.1,
                    maximum=10.0,
                    step=0.1,
                    value=5,
                )
                seed = gr.Slider(
                    label="种子(Seed)",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )
                randomize_seed = gr.Checkbox(label="随机种子(Randomize seed)", value=True)
        with gr.Column():
            gallery = gr.Gallery(label="生成的图像")
            usage_tips = gr.Markdown(label="PhotoMaker 的使用技巧", value=tips ,visible=True)

        files.upload(fn=swap_to_gallery, inputs=files, outputs=[uploaded_files, clear_button, files])
        remove_and_reupload.click(fn=remove_back_to_files, outputs=[uploaded_files, clear_button, files])

        submit.click(
            fn=remove_tips,
            outputs=usage_tips,            
        ).then(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=generate_image,
            inputs=[files, prompt, negative_prompt, style, num_steps, style_strength_ratio, num_outputs, guidance_scale, seed],
            outputs=[gallery, usage_tips]
        )

    gr.Examples(
        examples=get_example(),
        inputs=[files, prompt, style, negative_prompt],
        run_on_click=True,
        fn=upload_example_to_gallery,
        outputs=[uploaded_files, clear_button, files],
    )
    
    # gr.Markdown(article)
    
demo.launch(share=False)