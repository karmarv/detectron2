


#### Working with the Upstream GIT code [reference](https://stackoverflow.com/questions/7244321/how-do-i-update-a-github-forked-repository)

git clone git@github.com:vishwakarmarhl/detectron2.git
cd detectron2

"""

"""

1. Add the "upstream" to your cloned repository ("origin"): https://github.com/facebookresearch/detectron2
 ```git remote add upstream git@github.com:facebookresearch/detectron2.git```

2. Fetch the commits (and branches) from the "upstream":
 ```git fetch upstream ```

3. List and Switch to the "master" branch of your fork ("origin"):
 ```git branch -a```
 ```git checkout master ```

4. Stash the changes of your "master" branch:
 ```git stash ```

5. Merge the changes from the "master" branch of the "upstream" into your the "master" branch of your "origin":
 ```git merge upstream/master ```

6. Resolve merge conflicts if any and commit your merge
 ```git commit -am "Merged from upstream" ```

7. Push the changes to your fork
 ```git push ```

8. Get back your stashed changes (if any)
 ```git stash pop ```

9. You're done! Congratulations!

GitHub also provides instructions for this 