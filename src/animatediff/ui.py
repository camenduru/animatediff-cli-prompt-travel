import gradio as gr
from animatediff.front_utils import (get_schedulers, validate_inputs, getNow, download_video, create_file_list,
                                    find_safetensor_files, find_last_folder_and_mp4_file, find_next_available_number,
                                    find_and_get_composite_video, load_video_name, get_last_sorted_subfolder,
                                    create_config_by_gui, get_config_path, update_config, change_ip, change_ad, change_op,
                                    change_dp, get_first_sorted_subfolder, change_la)
from pathlib import Path


def launch():
    result_list = create_file_list("config/fix")
    safetensor_files = find_safetensor_files("data/sd_models")
    lora_files = find_safetensor_files("data/lora")
    mm_files = find_safetensor_files("data/motion_modules")
    schedulers = get_schedulers()
    ip_choice = ["full_face", "plus_face", "plus", "light"]
    ml_files = find_safetensor_files("data/motion_lora")
    
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
                        
                    with gr.Accordion("ControlNet", open=True):
                        ip_ch = gr.Checkbox(label="IPAdapter", value=False)
                        ip_image = gr.Image(height=256, type="pil", interactive=False)
                        # ip_upload = gr.UploadButton(label='Click to uplaod Image', file_types=["image"], file_count="single")
                        with gr.Row():
                            ip_scale = gr.Slider(minimum=0, maximum=2, step=0.1, value=0.5, label="IPAdapter scale", interactive=False)
                            ip_type = gr.Radio(choices=ip_choice, label="IPAdapter Type", value="plus_face", interactive=False)
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