#!/bin/bash

# List all fly apps and store them in an array
apps=($(flyctl apps list --json | jq -r '.[].Name'))

if [ ${#apps[@]} -eq 0 ]; then
    echo "No Fly.io applications found"
    exit 0
fi

echo "Found ${#apps[@]} applications"
echo "Applications to be destroyed:"
printf '%s\n' "${apps[@]}"

# Ask for confirmation
read -p "Are you sure you want to destroy all these applications? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    for app in "${apps[@]}"
    do
        echo "Destroying $app..."
        flyctl apps destroy "$app" --yes
    done
    echo "All applications have been destroyed"
else
    echo "Operation cancelled"
fi 