## Updating code
**from wilka's branch**
```
# step 1: commit your changes
git add -u; git add ${files}; git commit -m "name changes with something";

# step 2: checkout and pull from wilka's
git checkout nicegui; git pull

# step 3: go back to your branch and merge wilka's
git checkout samh; git merge nicegui
```

**pushing to wilka: do a pull request**
[github link](https://github.com/wcarvalho/human-dyna-web/compare/nicegui...samh?expand=1)
