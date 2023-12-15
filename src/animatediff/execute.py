import gradio as gr
import os
import glob
from datetime import datetime
import re
from pathlib import Path
import json
import yt_dlp
import shutil
import typer
from animatediff.stylize import create_config, create_mask, generate, composite
from animatediff.settings import ModelConfig, get_model_config
from animatediff.cli import refine

execute: typer.Typer = typer.Typer(
    name="execute",
    context_settings=dict(help_option_names=["-h", "--help"]),
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    help="execute video",
)

@execute.command(no_args_is_help=True)
def execute(
    videos: list = typer.Argument(..., help="List of video file paths"),
    configs: list = typer.Argument(..., help="List of config file paths"),
    urls: list = typer.Argument(..., help="List of URLs"),
    delete_if_exists: bool = typer.Option(False, "--deleteIfExists", help="Delete if files already exist"),
    is_test: bool = typer.Option(False, "--is_test", help="Run in test mode"),
    is_refine: bool = typer.Option(False, "--is_refine", help="Run in refine mode"),
    bg_config: str = typer.Option(None, help="Background prompt.json file path"),    
#    is_composite: bool = typer.Option(True, "--is_composite", help="Run in composite mode"),
):
    if videos:
        for video in videos:
            for config in configs:
                execute_impl(video=video, config=config, delete_if_exists=delete_if_exists, is_test=is_test, is_refine=is_refine, bg_config=bg_config)
    else:
        save_folder = './data/video'
        saved_files = download_videos(urls,save_folder)
        for saved_file in saved_files:
            print(saved_file)
            for config in configs:
                execute_impl(video=saved_file, config=config, delete_if_exists=delete_if_exists, is_test=is_test, is_refine=is_refine, bg_config=bg_config)

def execute_impl(video: str, config: str, delete_if_exists: bool, is_test: bool,is_refine: bool, bg_config: str):
    
    if video.startswith("/notebooks"):
        video = video[len("/notebooks"):]
    if config.startswith("/notebooks"):
        config = config[len("/notebooks"):]
    if bg_config is not None:
        if bg_config.startswith("/notebooks"):
            bg_config = bg_config[len("/notebooks"):]
    print(f"video1: {video}")
    yield 'generating config...', video, None, None, None
        
    video_name=video.rsplit('.', 1)[0].rsplit('/notebooks', 1)[-1].rsplit('/', 1)[-1]

    config = Path(config)
    model_config: ModelConfig = get_model_config(config)
    p_name = model_config.name
    
    if bg_config is not None:
        bg_config = Path(bg_config)
        bg_model_config: ModelConfig = get_model_config(bg_config)
    
#    stylize_dir='/storage/aj/animatediff-cli-prompt-travel/stylize/jjj-' + video_name
    stylize_dir='/storage/aj/animatediff-cli-prompt-travel/stylize/' + p_name + '-' + video_name
    stylize_fg_dir = stylize_dir + '/fg_00_'+p_name
    stylize_fg_dir = Path(stylize_fg_dir)
    stylize_bg_dir = stylize_dir + '/bg_'+p_name
    stylize_bg_dir = Path(stylize_bg_dir)
    stylize_dir = Path(stylize_dir)
    if stylize_dir.exists() and not delete_if_exists:
        print(f"config already exists. skip create-config")
    else:
        if stylize_dir.exists():
            print(f"Delete folder and create again")
            shutil.rmtree(stylize_dir)
        create_config(
            org_movie=video,
            config_org=config,
            fps=15,
        )
        create_mask(stylize_dir=stylize_dir, bg_config=bg_config, no_crop=True)
        
    yield 'generating fg bg video...', video, None, None, None
        
    if is_test:
        generate(stylize_dir=stylize_fg_dir, length=16)
        if bg_config is not None:
            generate(stylize_dir=stylize_bg_dir, length=16)
    else:
        generate(stylize_dir=stylize_fg_dir)
        if bg_config is not None:
            generate(stylize_dir=stylize_bg_dir)
            
    video2 = find_last_folder_and_mp4_file(stylize_fg_dir)
    print(f"video2: {video2}")
    if is_refine:
        yield 'refining fg video', video, video2, None, None
        
        result_dir = get_first_matching_folder(get_last_sorted_subfolder(stylize_fg_dir))
        refine(frames_dir=result_dir, out_dir=stylize_fg_dir, config_path=config, width=768)
#        !animatediff refine {result_dir} -W 768
    video3 = find_last_folder_and_mp4_file(stylize_fg_dir)
    print(f"video3: {video3}")
    yield 'compositing video', video2, video3, None

    fg_result = get_first_matching_folder(get_last_sorted_subfolder(stylize_fg_dir))
    bg_result = get_first_matching_folder(get_last_sorted_subfolder(stylize_bg_dir))

    if bg_config is not None:
        final_video_dir = composite(stylize_dir=stylize_dir, bg_dir=bg_result, fg_dir=fg_result)
    else:
        final_video_dir = composite(stylize_dir=stylize_dir, bg_dir=stylize_bg_dir/'00_img2img', fg_dir=fg_result)

    print(f"fg_フォルダ: {fg_result}")
    if bg_config is not None:
        print(f"bg_フォルダ: {bg_result}")
    else:
        print(f"bg_フォルダ: {stylize_bg_dir/'00_img2img'}")
    print(f"final_video_dir: {final_video_dir}")
    yield 'video is ready', video2, video3, final_video_dir

def find_last_folder_and_mp4_file(folder_path):
    # フォルダ内のフォルダを名前順にソート
    subfolders = sorted([f.path for f in os.scandir(folder_path) if f.is_dir()], key=lambda x: os.path.basename(x))

    # 一番最後のフォルダを取得
    last_folder = subfolders[-1]

    # 最後のフォルダ内の.mp4ファイルを検索
    mp4_files = glob.glob(os.path.join(last_folder, '*.mp4'))

    # 最初に見つかった.mp4ファイルのパスを取得して返却
    if mp4_files:
        return mp4_files[0]
    else:
        return None
    
def find_next_available_number(save_folder):
    existing_files = [f for f in os.listdir(save_folder) if f.startswith('dance') and f.endswith('.mp4')]
    existing_numbers = [int(file[5:10]) for file in existing_files]

    if existing_numbers:
        return max(existing_numbers) + 1
    else:
        return 1

def download_videos(video_urls, save_folder):
    saved_file_paths = []
    for video_url in video_urls:
        v_name = load_video_name(video_url)
        ydl_opts = {
            'outtmpl': os.path.join(save_folder, f'{v_name}.%(ext)s'),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(video_url, download=True)
            if 'entries' in result:
                for entry in result['entries']:
                    if 'filename' in entry:
                        saved_file_paths.append(entry['filename'])
                    else:
                        # Alternative approach to determine file name
                        file_extension = entry.get('ext', 'mp4')
                        saved_file_paths.append(os.path.join(save_folder, f'{v_name}.{file_extension}'))
            else:
                if 'filename' in result:
                    saved_file_paths.append(result['filename'])
                else:
                    # Alternative approach to determine file name
                    file_extension = result.get('ext', 'mp4')
                    saved_file_paths.append(os.path.join(save_folder, f'{v_name}.{file_extension}'))
    return saved_file_paths

def load_video_name(url):
    folder_path = './config/'
    file_path = os.path.join(folder_path, 'video_url.json')
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump([], file, ensure_ascii=False, indent=2)
        data = []
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

    existing_entry = next((entry for entry in data if entry['url'] == url), None)
    if existing_entry:
        return existing_entry['video_name']
    else:
        count = len(data) + 1
        new_video_name = f'dance{count:05d}'
        new_entry = {'url': url, 'video_name': new_video_name}
        data.append(new_entry)
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        return new_video_name

def get_last_sorted_subfolder(base_folder):
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
    sorted_subfolders = sorted(subfolders, key=lambda folder: os.path.basename(folder), reverse=True)
    last_sorted_subfolder = sorted_subfolders[0] if sorted_subfolders else None
    return last_sorted_subfolder

def get_first_matching_folder(base_folder):
    all_folders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
    pattern = re.compile(r'\d{2}-\d{4}-\d{2}-\d{2}_\d{2}-\d{2}')
    matching_folders = [folder for folder in all_folders if pattern.match(os.path.basename(folder))]
    first_matching_folder = matching_folders[0] if matching_folders else None
    return first_matching_folder

def get_first_matching_folder2(base_folder):
    all_folders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
    pattern = re.compile(r'\d{2}-\d{4}-\d{2}-\d{2}_\d{2}-\d{2}')
    matching_folders = [folder for folder in all_folders if pattern.match(os.path.basename(folder))]
    first_matching_folder = matching_folders[0] if matching_folders else None
    return first_matching_folder

def get_first_matching_refine_folder(base_folder):
    all_folders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
    pattern = re.compile(r'\d{2}-\d{4}-\d{2}-\d{2}_\d{2}-\d{2}')
    matching_folders = [folder for folder in all_folders if pattern.match(os.path.basename(folder))]
    first_matching_folder = matching_folders[0] if matching_folders else None
    return first_matching_folder