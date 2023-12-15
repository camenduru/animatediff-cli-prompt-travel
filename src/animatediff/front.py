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
        yield 'Error: Configs is required.', None

    if not urls:
        yield 'Error: URLs input is required.', None

    config_path = os.path.join("./config/fix", config)
    videos = []
    urls = [url.strip() for url in urls.split('\n') if url]
#    bg_config = bg_config.strip()
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
    
    # 指定されたフォルダ内のファイルを取得
    files = os.listdir(folder_path)
    
    # ファイルをフォルダ、拡張子を抜いたABC順にソート
    files.sort(key=lambda x: (os.path.splitext(x)[0].lower(), x))
    
    # ソートされたファイルのリストを作成
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            file_list.append(file_name)
    
    return file_list

from animatediff.cli import cli
def launch():
    cli()

    folder_path = "config/fix"
    # フォルダ内のファイルのリストを取得
    result_list = create_file_list(folder_path)
    iface = gr.Interface(
        fn=execute_wrapper, 
        inputs=[
            gr.Dropdown(choices=result_list, info="please select", label="Config"),
#            gr.Textbox(lines=3, placeholder="Enter URLs, separated by commas", label="URLs"),
            gr.Textbox(lines=1, placeholder="https://www.tiktok.com/@ai_hinahina/video/7308604819021827330", label="URL"),
            gr.Checkbox(label="Delete if exists"),
            gr.Checkbox(label="Is test", value=True),
            gr.Checkbox(label="Is refine"),
        ],
#        outputs=["label", "video"],
        outputs=[
            gr.Label(value="Status", label="Status", scale=3),
            gr.Video(width=256, title="Original Video", scale=1), 
            gr.Video(width=256, title="Front Video", scale=1), 
            gr.Video(width=256, title="Refined Front Video", scale=1), 
            gr.Video(width=256, title="Generated Video", scale=1),
        ],
#        capture_session=True,
        allow_flagging='never',
        title="AnimateDiff-GUI-prompt-travel",
    )
    iface.queue()
    iface.launch(share=True)
# def launch():
#     # 指定したフォルダ以下のファイル一覧を取得
#     folder_path = "/storage/aj/animatediff-cli-prompt-travel/config/fix"
#     config_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

#     # ファイル名から拡張子を取り除く
#     config_files_without_extension = [os.path.splitext(file)[0] for file in config_files]

#     # ファイル一覧をリストボックスの選択肢に変換
#     file_choices = [(file, file) for file in config_files_without_extension]


#     # Define the Gradio interface
#     iface = gr.Interface(
#         execute_wrapper, 
#         [
#     #        gr.Textbox(lines=3, placeholder="Enter video file paths, separated by commas", label="Videos"),
#             gr.CheckboxGroup(file_choices, info="please select", label="Configs"),
#     #        gr.Textbox(lines=3, placeholder="Enter config file paths, separated by commas", label="Configs"),
#             gr.Textbox(lines=3, placeholder="Enter URLs, separated by commas", label="URLs"),
#     #        gr.Textbox(lines=1, placeholder="Enter bg_config file path", label="BG Config"),
#             gr.Checkbox(label="Delete if exists"),
#             gr.Checkbox(label="Is test", value=True),
#             gr.Checkbox(label="Is refine"),
#         ],
#         outputs=["text", "video"],
#         allow_flagging='never',
#     )

#     # Launch the interface
#     iface.launch(share=True)

    # 無限ループを追加してセルが終了しないようにする
    while True:
        pass
