import gradio as gr
from animatediff.execute import execute, download_videos, execute_impl
import sys
import io
import os
import time

# Define the function signature
def execute_wrapper(config: str, urls: str, delete_if_exists: bool, is_test: bool, is_refine: bool):

    start_time = time.time()
    if not config:
        yield 'Error: Configs is required.', None, None, None, None
    if not urls:
        yield 'Error: URLs input is required.', None, None, None, None

    config_path = os.path.join("./config/fix", config)
    videos = []
    urls = [url.strip() for url in urls.split('\n') if url]
    bg_config = None
    yield 'generation Initiated...', None, None, None, None
    
    if videos:
        for video in videos:
            yield from execute_impl(video=video, config=config_path, delete_if_exists=delete_if_exists, is_test=is_test, is_refine=is_refine, bg_config=bg_config)
    else:
        save_folder = './data/video'
        saved_files = download_videos(urls,save_folder)
        for saved_file in saved_files:
            print(saved_file)
            print(config)
            yield from execute_impl(video=saved_file, config=config_path, delete_if_exists=delete_if_exists, is_test=is_test, is_refine=is_refine, bg_config=bg_config)    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"実行時間: {execution_time}秒")

def create_file_list(folder_path):
    file_list = []
    files = os.listdir(folder_path)
    files.sort(key=lambda x: (os.path.splitext(x)[0].lower(), x))
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            file_list.append(file_name)
    return file_list

def launch():

    folder_path = "config/fix"
    # フォルダ内のファイルのリストを取得
    result_list = create_file_list(folder_path)
#     iface = gr.Interface(
#         fn=execute_wrapper, 
#         inputs=[
#             gr.Textbox(lines=1, value="https://www.tiktok.com/@ai_hinahina/video/7312644055320513800" placeholder="https://www.tiktok.com/@ai_hinahina/video/7312644055320513800", label="URL"),
#             gr.Dropdown(choices=result_list, info="please select", label="Config"),
#             gr.Checkbox(label="Delete if exists"),
#             gr.Checkbox(label="Is test", value=True),
#             gr.Checkbox(label="Is refine"),
#         ],
# #        outputs=["label", "video"],
#         outputs=[
#             gr.Label(value="Status", label="Status"),
#             gr.Video(width=256, title="Original Video"), 
#             gr.Video(width=256, title="Front Video"), 
#             gr.Video(width=256, title="Refined Front Video"), 
#             gr.Video(width=256, title="Generated Video"),
#         ],
# #        capture_session=True,
#         allow_flagging='never',
#         title="AnimateDiff-GUI-prompt-travel",
#     )
#     iface.queue()
#     iface.launch(share=True)
    iface = gr.Interface(
        fn=execute_wrapper, 
        inputs=[
            gr.Textbox(lines=1, value="https://www.tiktok.com/@ai_hinahina/video/7312644055320513800" placeholder="https://www.tiktok.com/@ai_hinahina/video/7312644055320513800", label="URL"),
            gr.Dropdown(choices=result_list, info="please select", label="Config"),
            gr.Checkbox(label="Delete if exists"),
            gr.Checkbox(label="Is test", value=True),
            gr.Checkbox(label="Is refine"),
        ],
        outputs=[
            gr.Label(value="Status", label="Status"),
            gr.Video(width=256, title="Original Video"), 
            gr.Video(width=256, title="Front Video"), 
            gr.Video(width=256, title="Refined Front Video"), 
            gr.Video(width=256, title="Generated Video"),
        ],
        live=True,  # ライブモードを有効にする
        allow_flagging='never',
        title="AnimateDiff-GUI-prompt-travel",
    )
    iface.queue()
    iface.launch(share=True)
    # 無限ループを追加してセルが終了しないようにする
    while True:
        pass
