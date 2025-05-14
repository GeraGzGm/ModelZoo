#!/usr/bin/env python3
import os
import subprocess
import click

def get_completion_source(shell_type):
    """Efficiently get the completion script using subprocess."""
    env = os.environ.copy()
    env["_MODELZOO_COMPLETE"] = f"{shell_type}_source"
    return subprocess.check_output(["modelzoo"], env=env, text=True)

@click.command()
def install_completion():
    """Install shell completion for modelzoo"""
    shell = os.environ.get("SHELL", "")

    if "bash" in shell:
        completion = get_completion_source("bash")
        rc_file = os.path.expanduser("~/.bashrc")
        dest = os.path.expanduser("~/.modelzoo-complete.bash")
    elif "zsh" in shell:
        completion = get_completion_source("zsh")
        rc_file = os.path.expanduser("~/.zshrc")
        dest = os.path.expanduser("~/.modelzoo-complete.zsh")
    elif "fish" in shell:
        completion = get_completion_source("fish")
        rc_file = None
        dest = os.path.expanduser("~/.config/fish/completions/modelzoo.fish")
    else:
        raise click.ClickException("Unsupported shell. Only bash/zsh/fish are supported.")

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        f.write(completion)

    if rc_file:
        with open(rc_file, "a") as f:
            f.write(f"\n# ModelZoo completion\nsource {dest}\n")

    click.echo(f"✅ Completion installed for {shell}. Restart your shell to use it!")

if __name__ == "__main__":
    install_completion()
