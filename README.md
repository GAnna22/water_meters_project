## Проект по распознаванию показаний приборов учета (счетчиков воды)

Обработка реализуется несколькими шагами:
1) Ищем на фотографии прибор учета (ПУ) / счетчик (если их несколько, то запустится цикл и каждый будет обработан)
2) Вырезаем ПУ из фотографии и подаем на вход функции, которая рисует линии Хафа относительно горизонта (они нужны, чтобы довернуть ПУ на нужный градус, если фото повернуто)
3) Поворачиваем ПУ на найденный угол поворота + рассматриваем еще один вариант, когда фото повернуто на угол поворота + 180 градусов
4) Находим рамку с показаниями на приборе учета (на двух повернутых фотографиях)
4) Вырезаем на фото ПУ рамку с показаниями и подаем в модель распознавания цифр
5) Определяем местоположение запятой (считаем медианное значение в каждом из каналов RGB, а затем ищем разницу между ними для каждой найденной цифры; там, где наблюдается резкий скачок, нужно ставить разделитель целой и дробной части)
6) Если запятая найдена после 4 или 5 цифры, считая слева направо, мы ставим запятую, в противном случае не ставим


Для запуска проекта нужно:
1) в корне создать и активировать виртуальную среду командами 
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2) установить список необходимых для работы проекта библиотек командой
```bash
python3 -m pip install -r requirements.txt
```
3) перейти в папку src командой cd src/ и запустить модель командой 
```bash
streamlit run model.py --server.port=8501
```

Для переобучения или дообучения моделей можно воспользоваться ноутбуками в папке jupyter

Веса моделей: https://drive.google.com/drive/folders/1DssEfTVpBSP8XLZMKmQMQC6mLQvVI5Ot

Датасет: https://drive.google.com/drive/folders/18BUkhhlJMspmXEvc08MAYONUkrkR8qdW?usp=sharing