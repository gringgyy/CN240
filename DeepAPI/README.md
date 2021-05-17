# THIS IS A DEMO API #
# CONFIDENTIAL #

## Required Package

- fastapi
- uvicorn[standard]
- python-multipart
- numpy
- opencv-python


## How to use

  1. Clone this repo and open this repo directory.
```
git clone https://git.cils.space/pub/muteluh-fundus
cd muteluh-fundus
```

  2. Install required package.
```
pip install -r requirement.txt
```

  3. Start server.
```
uvicorn server:app --reload --host 0.0.0.0 --port 8000
``` 

  4. Open another terminal and run ssh tunnel (same working directory).
```
ssh -R [name]:80:localhost:8000 cn240@ondev.link -i ./cn240.key
```  