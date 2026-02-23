from modules.utils import run_command

def extract_clip(input_path, output_path, start, end):
    command = f"""
    ffmpeg -y -i {input_path} -ss {start} -to {end} -c copy {output_path}
    """
    run_command(command)