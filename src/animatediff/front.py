import gradio as gr
from animatediff.execute import execute
from animatediff.front_utils import (get_schedulers, validate_inputs, getNow, download_video, create_file_list,
                                    find_safetensor_files, find_last_folder_and_mp4_file, find_next_available_number,
                                    find_and_get_composite_video, load_video_name, get_last_sorted_subfolder,
                                    create_config_by_gui, get_config_path, update_config, change_ip, change_ad, change_op,
                                    change_dp, change_la, get_first_sorted_subfolder, get_stylize_dir, get_fg_dir,
                                    get_mask_dir, get_bg_dir)
from animatediff.settings import ModelConfig, get_model_config
from animatediff.video_utils import create_video
from animatediff.generate import save_output

from animatediff.stylize import generate, create_config, create_mask, composite
from animatediff.cli import refine



import io
import os
import time
from pathlib import Path
import shutil

# Define the function signature
def execute_wrapper(
      url: str, fps: int,
      inp_model: str, inp_mm: str,
      inp_sche: str, inp_step: int, inp_cfg: float, 
      inp_posi: str, inp_neg: str, 
      inp_lora1: str, inp_lora1_step: float,
      inp_lora2: str, inp_lora2_step: float,
      inp_lora3: str, inp_lora3_step: float,
      inp_lora4: str, inp_lora4_step: float,
      mo1_ch: str, mo1_scale: float,
      mo2_ch: str, mo2_scale: float,
      ip_ch: bool, ip_image: str, ip_scale: float, ip_type: str,
      mask_ch: str, mask_padding:int,
      ad_ch: bool, ad_scale: float, op_ch: bool, op_scale: float,
      dp_ch: bool, dp_scale: float, la_ch: bool, la_scale: float,
      delete_if_exists: bool, is_test: bool, is_refine: bool,
      progress=gr.Progress(track_tqdm=True)):
    
    yield 'generation Initiated...', None, None, None, None, None, None, None, gr.Button("Generating...", scale=1, interactive=False)
    start_time = time.time()

    time_str = getNow()
    validate_inputs(url)

    bg_config = None
    save_folder = 'data/video'
    saved_file = download_video(url, save_folder)
    video_name=saved_file.rsplit('.', 1)[0].rsplit('/notebooks', 1)[-1].rsplit('/', 1)[-1]
    stylize_dir= Path('/storage/aj/animatediff-cli-prompt-travel/stylize/' + video_name)
    create_config_by_gui(
        now_str=time_str,
        video = saved_file,
        stylize_dir = stylize_dir, 
        model=inp_model, motion_module=inp_mm, 
        scheduler=inp_sche, step=inp_step, cfg=inp_cfg, 
        head_prompt=inp_posi, neg_prompt=inp_neg,
        inp_lora1=inp_lora1, inp_lora1_step=inp_lora1_step,
        inp_lora2=inp_lora2, inp_lora2_step=inp_lora2_step,
        inp_lora3=inp_lora3, inp_lora3_step=inp_lora3_step,
        inp_lora4=inp_lora4, inp_lora4_step=inp_lora4_step,
        mo1_ch=mo1_ch, mo1_scale=mo1_scale,
        mo2_ch=mo2_ch, mo2_scale=mo2_scale,
        ip_ch=ip_ch, ip_image=ip_image, ip_scale=ip_scale, ip_type=ip_type,
        ad_ch=ad_ch, ad_scale=ad_scale, op_ch=op_ch, op_scale=op_scale,
        dp_ch=dp_ch, dp_scale=dp_scale, la_ch=la_ch, la_scale=la_scale,
    )

    yield from execute_impl(fps=fps,now_str=time_str,video=saved_file, delete_if_exists=delete_if_exists,
                            is_test=is_test, is_refine=is_refine, bg_config=bg_config, mask_ch=mask_ch, mask_padding=mask_padding)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"実行時間: {execution_time}秒")

def execute_impl(now_str:str, video: str, delete_if_exists: bool, is_test: bool, is_refine: bool, 
                 bg_config: str, fps:int, mask_ch:str, mask_padding:int):
    
    if video.startswith("/notebooks"):
        video = video[len("/notebooks"):]
    if bg_config is not None:
        if bg_config.startswith("/notebooks"):
            bg_config = bg_config[len("/notebooks"):]
    print(f"video1: {video}")

    mask_video = None
    depth_video = None
    lineart_video = None
    openpose_video = None
    front_video = None
    final_video = None
    
    yield 'generating config...', video, mask_video, depth_video, lineart_video, openpose_video, front_video, final_video, gr.Button("Generating...", scale=1, interactive=False)
    separator = os.path.sep
    video_name = os.path.splitext(os.path.normpath(video.replace('/notebooks', separator)))[0].rsplit(separator, 1)[-1]

    # video_name=video.rsplit('.', 1)[0].rsplit('/notebooks', 1)[-1].rsplit('/', 1)[-1]

    stylize_dir = get_stylize_dir(video_name)
    stylize_fg_dir = get_fg_dir(video_name)
    mask_dir = get_mask_dir(video_name)
    stylize_bg_dir = get_bg_dir(video_name)

    print(f"stylize_dir:{stylize_dir}")
    print(f"stylize_fg_dir:{stylize_fg_dir}")

    if bg_config is not None:
        bg_config = Path(bg_config)
        bg_model_config: ModelConfig = get_model_config(bg_config)

    if stylize_dir.exists() and not delete_if_exists:
        print(f"config already exists. skip create-config")
        if mask_ch != "As is Base":
            mask_video = mask_dir/'mask.mp4'
    else:
        if stylize_dir.exists():
            print(f"Delete folder and create again")
            shutil.rmtree(stylize_dir)
        create_config(org_movie=video, fps=15)
        # !animatediff stylize create-config {video} -f {fps}
    
    if not stylize_fg_dir.exists() and mask_ch != "As is Base":
        create_mask(stylize_dir=stylize_dir, bg_config=bg_config, no_crop=True)
        # !animatediff stylize create-mask {stylize_dir} -mp {mask_padding} -nc　
        if mask_ch:
            mask_video = mask_dir/'mask.mp4'
            save_output(
                None,
                mask_dir,
                mask_video,
                {"format":"mp4", "fps":fps},
                False,
                False,
                None,
            )
        
    update_config(now_str, video_name,mask_ch)
    config = get_config_path(now_str)
    model_config: ModelConfig = get_model_config(config)       
    
    yield 'generating fg bg video...', video, mask_video, depth_video, lineart_video, openpose_video, front_video, final_video, gr.Button("Generating...", scale=1, interactive=False)

    print(f"Start: stylize generate {stylize_fg_dir}")
    if is_test:
        if mask_ch != "As is Base":
            generate(stylize_dir=stylize_fg_dir, length=16)
            # !animatediff stylize generate {stylize_fg_dir} -L 16
            if bg_config is not None:
                generate(stylize_dir=stylize_bg_dir, length=16)
                # !animatediff stylize generate {stylize_bg_dir} -L 16
            front_video = find_last_folder_and_mp4_file(stylize_fg_dir)
            detect_map = get_last_sorted_subfolder(get_last_sorted_subfolder(stylize_fg_dir))
        else:
            generate(stylize_dir=stylize_dir, length=16)
            # !animatediff stylize generate {stylize_dir} -L 16
            front_video = find_last_folder_and_mp4_file(stylize_dir)
            detect_map = get_last_sorted_subfolder(get_last_sorted_subfolder(stylize_dir))
    else:
        if mask_ch != "As is Base":
            generate(stylize_dir=stylize_fg_dir)
            # !animatediff stylize generate {stylize_fg_dir}
            if bg_config is not None:
                generate(stylize_dir=stylize_bg_dir)
                # !animatediff stylize generate {stylize_bg_dir}
            front_video = find_last_folder_and_mp4_file(stylize_fg_dir)
            detect_map = get_last_sorted_subfolder(get_last_sorted_subfolder(stylize_fg_dir))
        else:
            print(f"generate {stylize_dir} start")
            generate(stylize_dir=stylize_dir)
            # !animatediff stylize generate {stylize_dir}
            front_video = find_last_folder_and_mp4_file(stylize_dir)
            detect_map = get_last_sorted_subfolder(get_last_sorted_subfolder(stylize_dir))
    print("###########################################################################################")
           
    print(f"video2: {front_video}")
    print(f"detect_map:{detect_map}")
    subfolders = [f.path for f in os.scandir(Path(detect_map)) if f.is_dir()]
    print(f"cn_folders: {subfolders}")
    for cn_folder in subfolders:
        # フォルダ名が "animate_diff" でない場合の処理
        print(f"cn_folder: {cn_folder}")
        if os.path.basename(cn_folder) != "animatediff_controlnet":
            filename = Path(cn_folder + '/' + os.path.basename(cn_folder)+'.mp4')
            save_output(
                None,
                Path(cn_folder),
                filename,
                {"format":"mp4","fps":fps},
                False,
                False,
                None,
            )

            if os.path.basename(cn_folder) == "controlnet_depth":
                depth_video = filename
            if os.path.basename(cn_folder) == "controlnet_lineart":
                lineart_video = filename
            if os.path.basename(cn_folder) == "controlnet_openpose":
                openpose_video = filename
    
    if is_refine:
        cur_width = model_config.stylize_config["0"]["width"]
        print(f"cur_width {cur_width}")
        new_width = int(float(cur_width) * float(1.5))
        print(f"refine width {new_width}")
        yield 'refining fg video', video, mask_video, depth_video, lineart_video, openpose_video, front_video, final_video, gr.Button("Generating...", scale=1, interactive=False)
        if mask_ch != "As is Base":
            result_dir = get_first_sorted_subfolder(get_last_sorted_subfolder(stylize_fg_dir))
            print(f"Start: Refine {result_dir} -width {new_width}")
            refine(frames_dir=result_dir, out_dir=stylize_fg_dir, config_path=config, width=new_width)
            # !animatediff refine {result_dir} -o {stylize_fg_dir} -c {config} -W 768
            front_video = find_last_folder_and_mp4_file(get_last_sorted_subfolder(stylize_fg_dir))
            print(f"video3: {front_video}")
            fg_result = get_first_sorted_subfolder(get_last_sorted_subfolder(get_last_sorted_subfolder(stylize_fg_dir)))
            print(f"aaaaaaaa{fg_result}")
            if mask_ch == 'Nothing Base':
                semi_final_video = find_last_folder_and_mp4_file(get_last_sorted_subfolder(stylize_fg_dir))
        else:
            result_dir = get_first_sorted_subfolder(get_last_sorted_subfolder(stylize_dir))
            print(f"Start: Refine {result_dir} -width {new_width}")
            refine(frames_dir=result_dir, out_dir=stylize_fg_dir, config_path=config, width=new_width)
            # !animatediff refine {result_dir} -o {stylize_dir} -c {config} -W 768
            front_video = find_last_folder_and_mp4_file(get_last_sorted_subfolder(stylize_fg_dir))
            print(f"video3: {front_video}")
            fg_result = get_last_sorted_subfolder(get_last_sorted_subfolder(get_last_sorted_subfolder(stylize_fg_dir)))
            semi_final_video = front_video
    else:
        if mask_ch != "As is Base":
            fg_result = get_first_sorted_subfolder(get_last_sorted_subfolder(stylize_fg_dir))
            if mask_ch == 'Nothing Base':
                semi_final_video = find_last_folder_and_mp4_file(stylize_fg_dir)
        else:
            fg_result = get_first_sorted_subfolder(get_last_sorted_subfolder(stylize_dir))
            semi_final_video = find_last_folder_and_mp4_file(stylize_dir)

    if mask_ch == "Original":
        yield 'composite video', video, mask_video, depth_video, lineart_video, openpose_video, front_video, final_video, gr.Button("Generating...", scale=1, interactive=False)
        bg_result = get_last_sorted_subfolder(stylize_bg_dir)

        print(f"fg_result:{fg_result}")
        if bg_config is not None:
            print(f"bg_dir: {bg_result}")
        else:
            print(f"bg_dir: {stylize_bg_dir/'00_img2img'}")

        if bg_config is not None:
            final_video_dir = composite(stylize_dir=stylize_dir, bg_dir=bg_result, fg_dir=fg_result)
            # !animatediff stylize composite {stylize_dir} -bg {bg_result} -fg {fg_result}  
        else:
            bg_result = stylize_bg_dir/'00_img2img'
            composite(stylize_dir=stylize_dir, bg_dir=stylize_bg_dir/'00_img2img', fg_dir=fg_result)
            # !animatediff stylize composite {stylize_dir} -bg {bg_result} -fg {fg_result}

        semi_final_video = find_and_get_composite_video(stylize_dir)

    print(f"final_video_dir: {semi_final_video}")

    final_dir = os.path.dirname(semi_final_video)
    final_video = os.path.join(final_dir,  video_name + ".mp4")
    
#    final_video_dir: stylize/dance00023/cp_2023-12-18_08-09/composite2023-12-18_08-09-41
    try:
        create_video(video, semi_final_video, final_video)
        print(f"new_file_path: {final_video}")
        
        yield 'video is ready!', video, mask_video, depth_video, lineart_video, openpose_video, front_video, final_video, gr.Button("Generate Video", scale=1, interactive=True)
    except Exception as e:
        print(f"error:{e}")
        final_video = semi_final_video
        yield 'video is ready!(no music added)', video, mask_video, depth_video, lineart_video, openpose_video, front_video, final_video, gr.Button("Generate Video", scale=1, interactive=True)
        
def launch():
    result_list = create_file_list("config/fix")
    safetensor_files = find_safetensor_files("data/sd_models")
    lora_files = find_safetensor_files("data/lora")
    mm_files = find_safetensor_files("data/motion_modules")
    schedulers = get_schedulers()
    ip_choice = ["full_face", "plus_face", "plus", "light"]
    ml_files = find_safetensor_files("data/motion_lora")
    bg_choice = ["Original", "Nothing Base", "As is Base"]
    
    with gr.Blocks() as iface:
        with gr.Row():
            gr.Markdown(
                """
                # AnimateDiff-V2V-GUI
                """, scale=8)
            btn = gr.Button("Generate Video", scale=1)
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    with gr.Group():
                        with gr.Row():
                            url = gr.Textbox(lines=1, value="https://www.tiktok.com/@ai_hinahina/video/7313863412541361426", placeholder="https://www.tiktok.com/@ai_hinahina/video/7313863412541361426", label="URL", scale=3)
                            fps = gr.Slider(minimum=8, maximum=64, step=1, value=16, label="fps", scale=1)
                    with gr.Group():
                        with gr.Row():
                            inp_model = gr.Dropdown(choices=safetensor_files, label="Model")
                            inp_mm = gr.Dropdown(choices=mm_files, label="Motion Module")
                    with gr.Group():
                        with gr.Row():
                            inp_sche = gr.Dropdown(choices=schedulers, label="Sampling Method")
                            inp_step = gr.Slider(minimum=1, maximum=20, step=1, value=10, label="Sampling Steps")
                            inp_cfg = gr.Slider(minimum=0.1, maximum=5, step=0.05,  value=2.4, label="CFG Scale")
                    inp_posi = gr.Textbox(lines=2, value="1girl, beautiful", placeholder="1girl, beautiful", label="Positive Prompt")
                    inp_neg = gr.Textbox(lines=2, value="low quality, low res,", placeholder="low quality, low res,", label="Negative Prompt")
                    with gr.Accordion("LoRAs", open=False):
                        with gr.Group():
                            with gr.Row():
                                inp_lora1 = gr.Dropdown(choices=lora_files, label="Lora1", scale=3)
                                inp_lora1_step = gr.Slider(minimum=0.1, maximum=3, step=0.05, value=1.0, label="LoRA1 Scale", scale=1)
                        with gr.Group():
                            with gr.Row():
                                inp_lora2 = gr.Dropdown(choices=lora_files, label="Lora2", scale=3)
                                inp_lora2_step = gr.Slider(minimum=0.1, maximum=3, step=0.05, value=1.0, label="LoRA2 Scale", scale=1)
                        with gr.Group():
                            with gr.Row():
                                inp_lora3 = gr.Dropdown(choices=lora_files, label="Lora3", scale=3)
                                inp_lora3_step = gr.Slider(minimum=0.1, maximum=3, step=0.05, value=1.0, label="LoRA3 Scale", scale=1)
                        with gr.Group():
                            with gr.Row():
                                inp_lora4 = gr.Dropdown(choices=lora_files, label="Lora4", scale=3)
                                inp_lora4_step = gr.Slider(minimum=0.1, maximum=3, step=0.05, value=1.0, label="LoRA4 Scale", scale=1)

                    with gr.Accordion("Motion Lora", open=False):
                        with gr.Row():
                            mo1_ch = gr.Dropdown(choices=ml_files, label="MotionLoRA1", scale=3)
                            mo1_scale = gr.Slider(minimum=0, maximum=2,  step=0.05, value=0.8, label="Motion LoRA1 scale")
                        with gr.Row():
                            mo2_ch = gr.Dropdown(choices=ml_files, label="MotionLoRA2", scale=3)
                            mo2_scale = gr.Slider(minimum=0, maximum=2,  step=0.05, value=0.8, label="Motion LoRA2 scale")
                        
                    with gr.Accordion("Effects", open=True):
                        ip_ch = gr.Checkbox(label="IPAdapter", value=False)
                        ip_image = gr.Image(height=256, type="pil", interactive=False)
                        # ip_upload = gr.UploadButton(label='Click to uplaod Image', file_types=["image"], file_count="single")
                        with gr.Row():
                            ip_scale = gr.Slider(minimum=0, maximum=2, step=0.1, value=0.5, label="IPAdapter scale", interactive=False)
                            ip_type = gr.Radio(choices=ip_choice, label="IPAdapter Type", value="plus_face", interactive=False)
                        with gr.Row():
                            mask_ch = gr.Radio(choices=bg_choice, label="Background Type", value="Original")
                            mask_padding = gr.Slider(minimum=-100, maximum=100, step=1, value=0, label="Mask Padding")
                        with gr.Row():
                            ad_ch = gr.Checkbox(label="AimateDiff Controlnet", value=True)
                            ad_scale = gr.Slider(minimum=0, maximum=2,  step=0.05, value=0.5, label="AnimateDiff Controlnet scale")
                        with gr.Row():
                            op_ch = gr.Checkbox(label="Open Pose", value=True)
                            op_scale = gr.Slider(minimum=0, maximum=2,  step=0.05, value=1.0, label="Open Pose scale")
                        with gr.Row():
                            dp_ch = gr.Checkbox(label="Depth", value=False)
                            dp_scale = gr.Slider(minimum=0, maximum=2,  step=0.05, value=1.0, label="Depth scale", interactive=False)
                        with gr.Row():
                            la_ch = gr.Checkbox(label="Lineart", value=False)
                            la_scale = gr.Slider(minimum=0, maximum=2,  step=0.05, value=1.0, label="Lineart scale", interactive=False)
                                
                 #   inp2 = gr.Dropdown(choices=result_list, info="please select", label="Config")
                    with gr.Row():
                        delete_if_exists = gr.Checkbox(label="Delete cache")
                        test_run = gr.Checkbox(label="Test Run", value=True)
                        refine = gr.Checkbox(label="Refine")
                    

            with gr.Column():
                with gr.Group():
                    o_status = gr.Label(value="Not Started Yet", label="Status", scale=2)
                    with gr.Row():
                        o_video1 = gr.Video(width=256, label="Original Video", scale=2)
                        with gr.Row():
                            o_video2_1 = gr.Video(width=128, label="Mask", scale=1)
                            o_video2_2 = gr.Video(width=128, label="Line Art", scale=1)
                        with gr.Row():
                            o_video2_3 = gr.Video(width=128, label="Depth", scale=1)
                            o_video2_4 = gr.Video(width=128, label="Open Pose", scale=1)
                    with gr.Row():
                        o_video3 = gr.Video(width=256, label="Front Video", scale=2)
                        o_video4 = gr.Video(width=256, label="Generated Video", scale=2)
        
        btn.click(fn=execute_wrapper,
                  inputs=[url, fps,
                          inp_model, inp_mm,
                          inp_sche, inp_step, inp_cfg, 
                          inp_posi, inp_neg, 
                          inp_lora1, inp_lora1_step,
                          inp_lora2, inp_lora2_step,
                          inp_lora3, inp_lora3_step,
                          inp_lora4, inp_lora4_step,
                          mo1_ch, mo1_scale,
                          mo2_ch, mo2_scale,
                          ip_ch, ip_image, ip_scale, ip_type,
                          mask_ch, mask_padding,
                          ad_ch, ad_scale, op_ch, op_scale,
                          dp_ch, dp_scale, la_ch, la_scale,
                          delete_if_exists, test_run, refine],
                  outputs=[o_status, o_video1, o_video2_1, o_video2_2, o_video2_3, o_video2_4, o_video3, o_video4, btn])

        ip_ch.change(fn=change_ip, inputs=[ip_ch], outputs=[ip_ch, ip_image, ip_scale, ip_type])        
        ad_ch.change(fn=change_ad, inputs=[ad_ch], outputs=[ad_ch, ad_scale])
        op_ch.change(fn=change_op, inputs=[op_ch], outputs=[op_ch, op_scale])
        dp_ch.change(fn=change_dp, inputs=[dp_ch], outputs=[dp_ch, dp_scale])
        la_ch.change(fn=change_la, inputs=[la_ch], outputs=[la_ch, la_scale])

        
    iface.queue()
    iface.launch(share=True)

    while True:
        pass

launch()


