import subprocess

def run_command(command):
    process = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if process.returncode != 0:
        raise Exception(process.stderr.decode())
    return process.stdout.decode()