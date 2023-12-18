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
import pytz
from animatediff.stylize import create_config, create_mask, generate, composite
from animatediff.settings import ModelConfig, get_model_config
from animatediff.cli import refine
from animatediff.video_utils import create_video

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
                if config.startswith("/notebooks"):
                    config = config[len("/notebooks"):]
                execute_impl(video=video, config=Path(config), delete_if_exists=delete_if_exists, is_test=is_test, is_refine=is_refine, bg_config=bg_config)
    else:
        save_folder = './data/video'
        saved_files = download_videos(urls,save_folder)
        for saved_file in saved_files:
            for config in configs:
                if config.startswith("/notebooks"):
                    config = config[len("/notebooks"):]
                execute_impl(video=saved_file, config=Path(config), delete_if_exists=delete_if_exists, is_test=is_test, is_refine=is_refine, bg_config=bg_config)

def execute_impl(video: str, config: Path, delete_if_exists: bool, is_test: bool,is_refine: bool, bg_config: str):
    print("impl")
    if video.startswith("/notebooks"):
        video = video[len("/notebooks"):]
    if bg_config is not None:
        if bg_config.startswith("/notebooks"):
            bg_config = bg_config[len("/notebooks"):]
    print(f"video1: {video}")
    # yield 'generating config...', video, None, None, None
        
    video_name=video.rsplit('.', 1)[0].rsplit('/notebooks', 1)[-1].rsplit('/', 1)[-1]

    model_config: ModelConfig = get_model_config(config)
    p_name = model_config.name
    
    if bg_config is not None:
        bg_config = Path(bg_config)
        bg_model_config: ModelConfig = get_model_config(bg_config)

#    stylize_dir='/storage/aj/animatediff-cli-prompt-travel/stylize/' + video_name
    stylize_dir='stylize/' + video_name
#    audio = stylize_dir + '/audio.wav'
    stylize_fg_dir = stylize_dir + '/fg_00_'+video_name
    stylize_fg_dir = Path(stylize_fg_dir)
    stylize_bg_dir = stylize_dir + '/bg_'+video_name
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
#        extract_audio(video, audio)

    # yield 'generating fg bg video...', video, None, None, None

    save_config_path = stylize_fg_dir.joinpath("prompt.json")
    save_config_path.write_text(model_config.json(indent=4), encoding="utf-8")
    
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
        # yield 'refining fg video', video, video2, None, None
        
        result_dir = get_first_matching_folder(get_last_sorted_subfolder(stylize_fg_dir))
        refine(frames_dir=result_dir, out_dir=stylize_fg_dir, config_path=config, width=768)
#        !animatediff refine {result_dir} -W 768
    video3 = find_last_folder_and_mp4_file(stylize_fg_dir)
    print(f"video3: {video3}")
    # yield 'compositing video',video, video2, video3, None

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

    # パスからフォルダを抽出
    final_dir = os.path.dirname(final_video_dir)
    # 新しいファイルのパスを作成
    new_file_path = os.path.join(final_dir,  p_name + ".mp4")
    
    cpmp4_file = str(final_video_dir) + '.mp4'
    
    print(f"final_video: {cpmp4_file}")
    
#   final_video_dir: stylize/dance00023/cp_2023-12-18_08-09/composite2023-12-18_08-09-41
    create_video(video, cpmp4_file, new_file_path)
    # yield 'video is ready', video2, video3, output_video_path
    
    
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

def download_video(video_url, save_folder) -> Path:
    v_name = load_video_name(video_url)
    ydl_opts = {
        'outtmpl': os.path.join(save_folder, f'{v_name}.%(ext)s'),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(video_url, download=True)
        if 'entries' in result:
            for entry in result['entries']:
                if 'filename' in entry:
                    return saved_file_paths
                else:
                    # Alternative approach to determine file name
                    file_extension = entry.get('ext', 'mp4')
                    return os.path.join(save_folder, f'{v_name}.{file_extension}')
        else:
            if 'filename' in result:
                return result['filename']
            else:
                # Alternative approach to determine file name
                file_extension = result.get('ext', 'mp4')
                return os.path.join(save_folder, f'{v_name}.{file_extension}')
    
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


def create_config_by_gui(
    now_str:str,
    video:str,
    stylize_dir: Path, 
    model: str, 
    motion_module: str, 
    scheduler: str, 
    step: int, 
    cfg: float, 
    head_prompt:str,
    neg_prompt:str,
    inp_lora1: str, inp_lora1_step: float,
    inp_lora2: str, inp_lora2_step: float,
    inp_lora3: str, inp_lora3_step: float,
    inp_lora4: str, inp_lora4_step: float,
) -> Path:
    
    config_org = "./config/fix/real_base2.json"
    model_config: ModelConfig = get_model_config(config_org)
    singapore_timezone = pytz.timezone('Asia/Singapore')
    time_str = datetime.now(singapore_timezone).strftime("%Y%m%d_%H%M")

    model_config.name = time_str
    model_config.path = Path(model)
    model_config.motion_module = Path(motion_module)
    model_config.steps = step
    model_config.guidance_scale = cfg
    model_config.scheduler = scheduler
    model_config.head_prompt = head_prompt
    model_config.n_prompt = [neg.strip() for neg in neg_prompt.split(',') if neg]
    model_config.stylize_config = {
            "original_video": {
                "path": video,
                "aspect_ratio": -1,
                "offset": 0
            },
            "create_mask": [
                "person"
            ],
            "composite": {
                "fg_list": [
                    {
                        "path": " absolute path to frame dir ",
                        "mask_path": " absolute path to mask dir (this is optional) ",
                        "mask_prompt": "person"
                    }
                ],
                "bg_frame_dir": "Absolute path to the BG frame directory",
                "hint": ""
            },
            "0": {
                "width": 512,
                "height": 904,
                "length": 140,
                "context": 16,
                "overlap": 4,
                "stride": 0
            }
        }
    model_config.lora_map = {}
    print(inp_lora1)
#    if inp_lora1 is not None:
    if len(inp_lora1) > 0:
        model_config.lora_map.update({inp_lora1[0] : {
            "region": ["0"],
            "scale": {"0": inp_lora1_step}
        }})
    if len(inp_lora2) > 0:
        model_config.lora_map.update({(inp_lora2[0],) : {
            "region": ["0"],
            "scale": {"0": inp_lora2_step}
        }})
    if len(inp_lora3) > 0:
        model_config.lora_map.update({(inp_lora3[0],) : {
            "region": ["0"],
            "scale": {"0": inp_lora3_step}
        }})
    if len(inp_lora4) > 0:
        model_config.lora_map.update({(inp_lora4[0],) : {
            "region": ["0"],
            "scale": {"0": inp_lora4_step}
        }})
    
    org_config_dir = Path("./config/from_ui")
    save_config_path = org_config_dir.joinpath(time_str+".json")
    save_config_path.write_text(model_config.json(indent=4), encoding="utf-8")
    
    return save_config_path
    
# class ModelConfig(BaseSettings):
#     name: str = Field(...)  # Config name, not actually used for much of anything
#     path: Path = Field(...)  # Path to the model
#     vae_path: str = ""  # Path to the model
#     motion_module: Path = Field(...)  # Path to the motion module
#     context_schedule: str = "uniform"
#     lcm_map: Dict[str,Any]= Field({})
#     gradual_latent_hires_fix_map: Dict[str,Any]= Field({})
#     compile: bool = Field(False)  # whether to compile the model with TorchDynamo
#     tensor_interpolation_slerp: bool = Field(True)
#     seed: list[int] = Field([])  # Seed(s) for the random number generators
#     scheduler: DiffusionScheduler = Field(DiffusionScheduler.k_dpmpp_2m)  # Scheduler to use
#     steps: int = 25  # Number of inference steps to run
#     guidance_scale: float = 7.5  # CFG scale to use
#     unet_batch_size: int = 1
#     clip_skip: int = 1  # skip the last N-1 layers of the CLIP text encoder
#     prompt_fixed_ratio: float = 0.5
#     head_prompt: str = ""
#     prompt_map: Dict[str,str]= Field({})
#     tail_prompt: str = ""
#     n_prompt: list[str] = Field([])  # Anti-prompt(s) to use
#     is_single_prompt_mode : bool = Field(False)
#     lora_map: Dict[str,Any]= Field({})
#     motion_lora_map: Dict[str,float]= Field({})
#     ip_adapter_map: Dict[str,Any]= Field({})
#     img2img_map: Dict[str,Any]= Field({})
#     region_map: Dict[str,Any]= Field({})
#     controlnet_map: Dict[str,Any]= Field({})
#     upscale_config: Dict[str,Any]= Field({})
#     stylize_config: Dict[str,Any]= Field({})
#     output: Dict[str,Any]= Field({})
#     result: Dict[str,Any]= Field({})
