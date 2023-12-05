import typer
import gradio as gr

app = typer.Typer()

@app.command(no_args_is_help=True)
def auto_exec(
    video_url: str = typer.Argument(..., help="Video URL"),
    video: str = typer.Argument(..., help="Video file path"),
    config: str = typer.Argument(..., help="Config file path"),
    delete_if_exists: bool = typer.Option(False, "--deleteIfExists", help="Delete if files already exist"),
    is_test: bool = typer.Option(False, "--is_test", help="Run in test mode"),
    is_refinewo: bool = typer.Option(False, "--is_refinewo", help="Run in refinewo mode"),
):
    # ここで引数を使用して AnimateDiff を auto-exec する処理を書く
    typer.echo(f"Video URL: {video_url}")
    typer.echo(f"Video File: {video}")
    typer.echo(f"Config File: {config}")
    typer.echo(f"Delete if exists: {delete_if_exists}")
    typer.echo(f"Is Test: {is_test}")
    typer.echo(f"Is Refinewo: {is_refinewo}")
    # AnimateDiff の auto-exec の処理を追加

    #VideoNameの引数でそこからvideo_nameを取得するロジックをここに追加する
    print("video name:", video)

    if video.startswith("/notebooks"):
        video = video[len("/notebooks"):]

    video_name=video.rsplit('.', 1)[0].rsplit('/notebooks', 1)[-1].rsplit('/', 1)[-1]

    stylize_dir='/storage/aj/animatediff-cli-prompt-travel/stylize/jjj-' + video_name
    stylize_fg_dir = stylize_dir + '/fg_00_jjj'
    stylize_bf_dir = stylize_dir + '/bg_jjj'
    path_to_check = Path(stylize_dir)
    
    if path_to_check.exists() and not deleteIfExists:
        print(f"config already exists. skip create-config")
    else: path_to_check.exists() and deleteIfExists:
        try:
            print(f"Delete folder and create again")
            shutil.rmtree(directory_path)
        except Exception as e:
            print(f"no folder exists")
#        !rm -r {stylize_dir}
#        !animatediff stylize create-config {video} -c {con} -f 15
        create_config(
            org_movie=video,
            config_org=con,
            fps=15,
        )
        create_mask(stylize_dir)
#        !animatediff stylize create-mask {stylize_dir} 

    if is_test:
        generate(stylize_dir=stylize_fg_dir, length=16)
#        !animatediff stylize generate {stylize_fg_dir} -L 16
    else:
        generate(stylize_dir=stylize_fg_dir)
#        !animatediff stylize generate {stylize_fg_dir}

    if is_refine:
        result_dir = get_first_matching_folder(get_last_sorted_subfolder(stylize_fg_dir))
        refine(out_dir=stylize_fg_dir, config_path=config_path, width=768)
#        !animatediff refine {result_dir} -W 768  

def find_next_available_number(save_folder):
    existing_files = [f for f in os.listdir(save_folder) if f.startswith('dance') and f.endswith('.mp4')]
    existing_numbers = [int(file[5:10]) for file in existing_files]

    if existing_numbers:
        return max(existing_numbers) + 1
    else:
        return 1

def download_videos(video_urls, save_folder):
#    next_available_number = find_next_available_number(save_folder)
    v_name = load_video_name(video_urls)
    ydl_opts = {
        'outtmpl': os.path.join(save_folder, f'{v_name}.%(ext)s'),
    }
    saved_file_paths = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for video_url in video_urls:
            result = ydl.extract_info(video_url, download=True)
            if 'entries' in result:
                for entry in result['entries']:
                    if 'filename' in entry:
                        saved_file_paths.append(entry['filename'])
                    else:
                        # Alternative approach to determine file name
                        file_extension = entry.get('ext', 'mp4')
                        saved_file_paths.append(os.path.join(save_folder, f'dance{next_available_number:05d}.{file_extension}'))
                    next_available_number += 1
            else:
                if 'filename' in result:
                    saved_file_paths.append(result['filename'])
                else:
                    # Alternative approach to determine file name
                    file_extension = result.get('ext', 'mp4')
                    saved_file_paths.append(os.path.join(save_folder, f'dance{next_available_number:05d}.{file_extension}'))
                    next_available_number += 1
    return saved_file_paths

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

def load_video_name(url, video_name):
    folder_path = '/config/'
    file_path = os.path.join(folder_path, 'video_url.json')
    if not os.path.exists(file_path):
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

if videos:
    for video in videos:
        for con in configs:
            try:
                exec_video(video, con)
            except:
                print("An exception occurred")
else:
    save_folder = '/storage/aj/animatediff-cli-prompt-travel/data/video'
    saved_files = download_videos(video_urls,save_folder)
    for saved_file in saved_files:
        print(saved_file)
        for con in configs:
            try:
                exec_video(saved_file, con)
            except:
                print("An exception occurred")
    
