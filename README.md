## prestart
```
pip install -r requirements.txt
```

## server
```
make clean
make start
```

## client
```
curl -X POST -F "file=@1.mat" "http://localhost:5000?ll=5"
```
