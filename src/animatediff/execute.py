import os
import re
from pathlib import Path
import json
import yt_dlp
import shutil
import typer
from animatediff.stylize import create_config, create_mask, generate
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
    bg_config: str = typer.Argument(..., help="Background prompt.json file path"),
    urls: list = typer.Argument(..., help="List of URLs"),
    delete_if_exists: bool = typer.Option(False, "--deleteIfExists", help="Delete if files already exist"),
    is_test: bool = typer.Option(False, "--is_test", help="Run in test mode"),
    is_refine: bool = typer.Option(False, "--is_refinewo", help="Run in refinewo mode"),
    
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
    if bg_config.startswith("/notebooks"):
        bg_config = bg_config[len("/notebooks"):]
        
    video_name=video.rsplit('.', 1)[0].rsplit('/notebooks', 1)[-1].rsplit('/', 1)[-1]

    config = Path(config)
    model_config: ModelConfig = get_model_config(config)
    p_name = model_config.name

    bg_config = Path(bg_config)
    bg_model_config: ModelConfig = get_model_config(bg_config)
    
#    stylize_dir='/storage/aj/animatediff-cli-prompt-travel/stylize/jjj-' + video_name
    stylize_dir='./stylize/' + p_name + '-' + video_name
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
        create_mask(stylize_dir=stylize_dir, bg_config=bg_config)
#    !animatediff stylize create-mask {stylize_dir}
    if is_test:
        generate(stylize_dir=stylize_fg_dir, length=16)
        generate(stylize_dir=stylize_bg_dir, length=16)

#        !animatediff stylize generate {stylize_fg_dir} -L 16
    else:
        generate(stylize_dir=stylize_fg_dir)
        generate(stylize_dir=stylize_bg_dir)
#        !animatediff stylize generate {stylize_fg_dir}

    if is_refine:
        result_dir = get_first_matching_folder(get_last_sorted_subfolder(stylize_fg_dir))
        refine(frames_dir=result_dir, out_dir=stylize_fg_dir, config_path=config, width=768)
#        !animatediff refine {result_dir} -W 768

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
