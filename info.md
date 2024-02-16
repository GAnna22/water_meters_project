# multipage-test

# How to launch

docker build -t ocr-ui .
docker run --mount type=bind,source="$(pwd)"/data,target=/app/data -it --rm -p 8503:8503 ocr-ui
