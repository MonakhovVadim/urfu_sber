# Инструкция по использованию приложения
Приложение производит оценку технологического риска внедрения релиза с использованием математической оценки (на основании подготовленной скоринговой модели), а также с использованием предсказания обученной модели.
Оба предсказания выводятся пользователю на главной странице приложения после того, как он укажет оценку каждому критерию риска и нажмет кнопку Определить риск.

![app_screenshot.png](app_screenshot.png)

## Математическая модель оценки риска
Оценка производится на основании скоринговой карты, представляющей из себя список критериев оценки риска, их весового коэффициента, а также минимального и максимального значения (целые числа), который может получить критерий. Помимо этого, указаны описания критериев и обоснования для предлагаемого веса критерия. 
Веса всех критериев могут быть подобраны таким образом, чтобы их сумма составляла 1. Однако это не обязательно, поскольку в любом случае производится автоматический пересчет и нормализация весов для их приведения к 1 (100%). Тем самым пользователь (эксперт), который будет настраивать скоринговую модель «под себя», может быть удобную для себя подход при определении весов. 
В рамках проекта, нами создана дефолтная скоринговая модель, которая основана на нашем представлении о значимости тех или иных критериев при оценке риска внедрения и предоставлены наши аргументы для каждого веса. Однако мы глубоко убеждены в том, что установление весов должно осуществляться либо на основании истории внедрения релизов в конкретной компании (например, путем анализа реального датасета с этими данными вручную или ML моделью), либо экспертом этой компании. Т.е. мы считаем, что в разных компаниях могут превалировать разные критерии в итоговой оценке риска. 
Скоринговая карта, с дефолтными значениями весов, представляет собой Excel документ. При необходимости, в него можно внести новые критерии, задать вес и критерий будет добавлен в риск-оценку. 
Также данная скоринговая карта является основой для математической оценки риска внедрения. Веса критериям можно задать на отдельной странице приложения. Предполагается, что эксперт компании, в которой будет использоваться решение, сможем сам донастроить веса критериям. 
Формула расчета риска внедрения:

$risk = Σ criteria * weight$, 

где 
 - $criteria$ – это критерий риска, который оценивает пользователь
 - $weight$ – вес критерия

Настройка весов критериев производится на странице *Настройка скоркарты*. Столбец «weight» таблицы является редактируемым. При нажатии на кнопку сохранить изменения, скоркарта изменится и при определении риска алгоритмическим методом, будут использоваться новые веса. 

## Оценка риска ML моделью
Модель оценки риска обучена на синтетическом датасете, который генерируется на основании дефолтной скоркарты в рамках автоматического pipeline сборки Docker контейнера. Пользователь имеет возможность загрузить собственный датасет с критериями и их оценками, а также таргетами. Однако столбцы датасета должны соответствовать критериям, указанным в дефолтном датасете.
Если критерии (фичи) в датасете пользователя не соответствуют тем, что уже реализованы в интерфейсе, то прежде, чем обучать модель, следует изменить критерии в дефолтной скоркарте. В нее необходимо внести те критерии, которые планируется использовать для обучения и пересобрать приложение, поскольку оно отображает только те критерии оценки риска, которые указаны в дефолтной скоркарте.
При загрузке датасета модель проводит препроцессинг данных, нормализацию, сохраняет pipeline предобработки и после этого обучается с помощью линейной регрессии библиотеки scikit-learn. В следующем релизе планируется реализовать обучение нескольких ML моделей с автоматическим выбором и сохранением той модели, которая показывает наилучшие метрики. 
После обучения модели, на главное странице приложения можно получить предсказание оценки риска внедрения.
Заметим, что при генерации **синтетического** датасета, оценки критериев генерируются рандомно, а для расчета таргета используется математический алгоритм (из скоркарты).
Соответственно модель, обученная на таком датасете (происходит при первом запуске приложения), предсказывает оценку риска равную математическому расчету оценки риска. Это связано с тем, что в процессе обучения модель находит оптимальные веса, которые, собственно, совпадают с весами дефолтной скоркарты.

# Сценарии использования приложения
Приложение позволяет рассчитывать оценку риска внедрения математически на основании заданных весов и предсказанием модели, что позволяет подходить к оценке риска максимально гибко. Так, оценка моделью требует наличия датасета с реальными данными и проведенными релизами и имеет некоторый запаздывающий эффект во времени (за время создания датасета структура рисков и их воздействие на риск могли существенно измениться).
Оценка же весов экспертом позволяет более чутко и оперативно реагировать на изменение структуры рисков и их значимость. Однако любой эксперт – это всегда человеческий фактор и доля субъективизма. 
Таким образом, пользователь может оценить плюсы и минусы обоих подходов к оценке и выбрать тот, которые окажется более эффективным в его ситуации. Либо может усреднять обе оценки и получать оценку, которая обладает плюсами и минусами обеих систем оценки. 