import subprocess
import argparse

def launch_experiment(name, env_vars):
    # Construct the flyctl launch command
    launch_cmd = [
        "flyctl", "launch",
        "--dockerfile", "Dockerfile",
        "--name", f"human-dyna-{name}",
        "--config", f"configs/human-dyna-{name}.toml",
        "--vm-size", "performance-2x"
    ]
    launch_cmd.extend(env_vars)

    # Run the flyctl launch command
    subprocess.run(launch_cmd, check=True)

    # Deploy the website
    deploy_cmd = ["flyctl", "deploy", "--config", f"configs/human-dyna-{name}.toml"]
    subprocess.run(deploy_cmd, check=True)

    # Scale the application
    scale_cmd = [
        "flyctl", "scale", "count", "4",
        "--config", f"configs/human-dyna-{name}.toml",
        "--region", "iad,sea,lax,den"
    ]
    subprocess.run(scale_cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch a Fly.io experiment")
    parser.add_argument("name", help="Name of the experiment")
    parser.add_argument("--env", action="append", help="Environment variables in the format KEY=VALUE")
    
    args = parser.parse_args()

    env_vars = ["--env", args.name]
    if args.env:
        for env in args.env:
            env_vars.extend(["--env", env])
    
    launch_experiment(args.name, env_vars)