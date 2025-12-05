### Installation and running guide

1. Activate virtual environment:

```
python3 -m venv env
```

2. Install required packages (find the requirements.txt in the root of the project):
```
pip3 install -r requirements.txt
```

3. Create folder 'input' and 'output':
```
mkdir input
mkdir output
```
4. Put images in 'input' folder and run 'resizer.py'.
5. Expect resized images in 'output' folder.

### Customization

Optionally, you can you flags when running the script to adapt it for your needs:
```
python3 resizer.py -height 100 -width 200 -input "./input1/" -output "./my_output/"
```


