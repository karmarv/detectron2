
#### A. Working with [PyTorch](https://pytorch.org/get-started/locally/) and [CUDA 10.1](https://www.tensorflow.org/install/gpu#install_cuda_with_apt)

```

```

#### B. Python [Virtual Environment Wrapper](https://medium.com/the-andela-way/configuring-python-environment-with-virtualenvwrapper-8745c2895745)

``` 
 mkvirtualenv det2 -p python3 
 pip install -r requirements
```

#### C. Working with the Upstream GIT code [reference](https://stackoverflow.com/questions/7244321/how-do-i-update-a-github-forked-repository)

```
git clone https://github.com/vishwakarmarhl/detectron2
cd detectron2
```

1. Add the "upstream" to your cloned repository ("origin"):
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