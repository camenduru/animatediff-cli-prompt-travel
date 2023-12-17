import gradio as gr
from animatediff.execute import execute, download_video, execute_impl, create_config_by_gui
import sys
import io
import os
import time
import pytz
from pathlib import Path
from datetime import datetime

# Define the function signature
def execute_wrapper(
      url: str, 
      inp_model: str, inp_mm: str,
      inp_sche: str, inp_step: int, inp_cfg: float, 
      inp_posi: str, inp_neg: str, 
      inp_lora1: str, inp_lora1_step: float,
      inp_lora2: str, inp_lora2_step: float,
      inp_lora3: str, inp_lora3_step: float,
      inp_lora4: str, inp_lora4_step: float,
      delete_if_exists: bool, is_test: bool, is_refine: bool):

    start_time = time.time()
    singapore_timezone = pytz.timezone('Asia/Singapore')
    time_str = datetime.now(singapore_timezone).strftime("%Y%m%d_%H%M")

    if not url:
        yield 'Error: URLs input is required.', None, None, None, None
        return
    bg_config = None
    yield 'generation Initiated...', None, None, None, None
    
    save_folder = 'data/video'
    saved_file = download_video(url, save_folder)
    video_name=saved_file.rsplit('.', 1)[0].rsplit('/notebooks', 1)[-1].rsplit('/', 1)[-1]
#    stylize_dir= Path('/storage/aj/animatediff-cli-prompt-travel/stylize/' + time_str + '-' + video_name)
    stylize_dir= Path('/storage/aj/animatediff-cli-prompt-travel/stylize/' + video_name)
    config_path = create_config_by_gui(
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
    )

    print(config_path)
#    yield from execute_impl(video=saved_file, config=config_path, delete_if_exists=delete_if_exists, is_test=is_test, is_refine=is_refine, bg_config=bg_config)    
    execute_impl(video=saved_file, config=config_path, delete_if_exists=delete_if_exists, is_test=is_test, is_refine=is_refine, bg_config=bg_config)    

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"実行時間: {execution_time}秒")


def launch():
    folder_path = "config/fix"
    result_list = create_file_list(folder_path)

    model_folder = "data/sd_models"
    safetensor_files = find_safetensor_files(model_folder)

    lora_folder = "data/lora"
    lora_files = find_safetensor_files(lora_folder)

    mm_folder = "data/motion_modules"
    mm_files = find_safetensor_files(mm_folder)
    
    schedulers = [("LCM", "lcm"),
                  ("DDIM", "ddim"),
                  ("PNDM", "pndm"),
                  ("Heun", "heun"),
                  ("UniPC", "unipc"),
                  ("Euler", "euler"),
                  ("Euler a", "euler_a"),
                  ("LMS", "lms"),
                  ("LMS Karras", "k_lms"),
                  ("DPM2", "dpm_2"),
                  ("DPM2 Karras", "k_dpm_2"),
                  ("DPM2 a", "dpm_2_a"),
                  ("DPM2 a Karras", "k_dpm_2_a"),
                  ("DPM++ 2M", "dpmpp_2m"),
                  ("DPM++ 2M Karras", "k_dpmpp_2m"),
                  ("DPM++ SDE", "dpmpp_sde"),
                  ("DPM++ SDE Karras", "k_dpmpp_sde"),
                  ("DPM++ 2M SDE", "dpmpp_2m_sde"),
                  ("DPM++ 2M SDE Karras", "k_dpmpp_2m_sde")]
    
    ip_choice = ["full_face", "plus_face", "plus", "light"]
#    with gr.Blocks(css=""".gradio-container {margin: 0 !important; padding: 5px !important};""") as iface:
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
                    url = gr.Textbox(lines=1, value="https://www.tiktok.com/@ai_hinahina/video/7312648648553204999", placeholder="https://www.tiktok.com/@ai_hinahina/video/7312648648553204999", label="URL")
                    with gr.Group():
                        with gr.Row():
                            inp_model = gr.Dropdown(choices=safetensor_files, label="Model")
                            inp_mm = gr.Dropdown(choices=mm_files, label="Motion Module")
                    with gr.Group():
                        with gr.Row():
                            inp_sche = gr.Dropdown(choices=schedulers, label="Sampling Method")
                            inp_step = gr.Slider(minimum=1, maximum=20, step=1, value=8, label="Sampling Steps")
                            inp_cfg = gr.Slider(minimum=0.1, maximum=10, step=0.1,  value=2.3, label="CFG Scale")
                    inp_posi = gr.Textbox(lines=2, value="1girl, beautiful", placeholder="1girl, beautiful", label="Positive Prompt")
                    inp_neg = gr.Textbox(lines=2, value="low quality, low res,", placeholder="low quality, low res,", label="Negative Prompt")
                    with gr.Accordion("LoRAs", open=False):
                        with gr.Group():
                            with gr.Row():
                                inp_lora1 = gr.Dropdown(choices=lora_files, label="Lora1", scale=3)
                                inp_lora1_step = gr.Slider(minimum=0.1, maximum=3, step=0.1, value=1.0, label="Scale", scale=1)
                        with gr.Group():
                            with gr.Row():
                                inp_lora2 = gr.Dropdown(choices=lora_files, label="Lora2", scale=3)
                                inp_lora2_step = gr.Slider(minimum=0.1, maximum=3, step=0.1, value=1.0, label="Scale", scale=1)
                        with gr.Group():
                            with gr.Row():
                                inp_lora3 = gr.Dropdown(choices=lora_files, label="Lora3", scale=3)
                                inp_lora3_step = gr.Slider(minimum=0.1, maximum=3, step=0.1, value=1.0, label="Scale", scale=1)
                        with gr.Group():
                            with gr.Row():
                                inp_lora4 = gr.Dropdown(choices=lora_files, label="Lora4", scale=3)
                                inp_lora4_step = gr.Slider(minimum=0.1, maximum=3, step=0.1, value=1.0, label="Scale", scale=1)

                    with gr.Accordion("ControlNet", open=False):
                        with gr.Group():
                            ip_ch = gr.Checkbox(label="Enable IPAdapter", value=False),
                            ip_image = gr.Image()
                            with gr.Row():
                                ip_scale = gr.Slider(minimum=0, maximum=3, step=0.1, value=1.0, label="scale")
                                ip_type = gr.Radio(choices=ip_choice, label="Type")
                        with gr.Group():
                            with gr.Row():
                                ad_ch = gr.Checkbox(label="Enable AnimateDiff Controlnet", value=True),
                                ad_scale = gr.Slider(minimum=0, maximum=3,  step=0.1, value=1.0, label="scale")
                        with gr.Group():
                            with gr.Row():
                                op_ch = gr.Checkbox(label="Enable Open Pose", value=True),
                                op_scale = gr.Slider(minimum=0, maximum=3,  step=0.1, value=1.0, label="scale")
                        with gr.Group():
                            with gr.Row():
                                dp_ch = gr.Checkbox(label="Enable Depth", value=False),
                                dp_scale = gr.Slider(minimum=0, maximum=3,  step=0.1, value=1.0, label="scale")
                        with gr.Group():
                            with gr.Row():
                                la_ch = gr.Checkbox(label="Enable Lineart", value=False),
                                la_scale = gr.Slider(minimum=0, maximum=3,  step=0.1, value=1.0, label="scale")
                                
                 #   inp2 = gr.Dropdown(choices=result_list, info="please select", label="Config")
                    with gr.Row():
                        delete_if_exists = gr.Checkbox(label="Delete cache")
                        test_run = gr.Checkbox(label="Test Run", value=True)
                        refine = gr.Checkbox(label="Refine")
                    

            with gr.Column():
                with gr.Group():
                    out1 = gr.Label(value="Not Started Yet", label="Status")
                    with gr.Row():
                        out2 = gr.Video(width=256, label="Original Video")
                        out3 = gr.Video(width=256, label="Front Video")
                    with gr.Row():
                        out4 = gr.Video(width=256, label="Refined Front Video")
                        out5 = gr.Video(width=256, label="Generated Video")

        btn.click(fn=execute_wrapper,
                  inputs=[url, 
                          inp_model, inp_mm,
                          inp_sche, inp_step, inp_cfg, 
                          inp_posi, inp_neg, 
                          inp_lora1, inp_lora1_step,
                          inp_lora2, inp_lora2_step,
                          inp_lora3, inp_lora3_step,
                          inp_lora4, inp_lora4_step,
                          delete_if_exists, test_run, refine],
                  outputs=[out1, out2, out3, out4, out5])

        # ip_ch.change(fn=change_ip,
        #           inputs=[ip_ch],
        #           outputs=[ip_ch, ip_image, ip_scale, ip_type])
        # ad_ch.change(fn=change_ad,
        #           inputs=[ad_ch],
        #           outputs=[ad_ch, ad_scale])
        # op_ch.change(fn=change_op,
        #           inputs=[op_ch],
        #           outputs=[op_ch, op_scale])
        # dp_ch.change(fn=change_dp,
        #           inputs=[dp_ch],
        #           outputs=[dp_ch, dp_scale])
        # la_ch.change(fn=change_la,
        #           inputs=[la_ch],
        #           outputs=[la_ch, la_scale])

        
    iface.queue()
    iface.launch(share=True)

    # 無限ループを追加してセルが終了しないようにする
    while True:
        pass
    
def create_file_list(folder_path):
    file_list = []
    files = os.listdir(folder_path)
    files.sort(key=lambda x: (os.path.splitext(x)[0].lower(), x))
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            file_list.append(file_name)
    return file_list


def find_safetensor_files(folder, suffix=''):
    result_list = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".safetensors"):
                file_path = os.path.join(root, file)
                folder_name = os.path.relpath(root, folder)
                file_name = os.path.splitext(file)[0]
                
                if folder_name != ".":
                    file_name = os.path.join(folder_name, file_name)
                
                result_name = f"{suffix}{file_name}"
                result_path = os.path.relpath(file_path, folder)
                if folder.startswith("data/"):
                    folder2 = folder[len("data/"):]
                result_list.append((result_name, folder2+'/'+result_path))

        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            subdir_suffix = f"{suffix}{subdir}/" if suffix else f"{subdir}/"
            result_list.extend(find_safetensor_files(subdir_path, subdir_suffix))
            
    result_list.sort(key=lambda x: x[0])  # file_name でソート
    return result_list