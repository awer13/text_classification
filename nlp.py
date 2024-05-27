import streamlit as st
import joblib
from streamlit import session_state as _state
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def main():
    # Register your pages
    pages = {
        "Главная страница": page_first,
        "Загрузка датасета": page_second,
        "Обработка и визуализация датасета": page_third,
        "Тренировка и проверка модели": page_forth
    }

    st.sidebar.title("Веб-ресурстардағы киберқауіпті мәтіндерді анықтау")

    # Widget to select your page, you can choose between radio buttons or a selectbox
    page = st.sidebar.radio("Выберите подходящую страницу", tuple(pages.keys()))
    # page = st.sidebar.radio("Select your page", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page]()


def page_first():
    st.header(
        """Бұл жұмыста киберқауіпті мәтін ретінде `эйджизм`, `эйблизм` және `адамды әлеуметтік кемсітуге` қатысты мәтіндер қарастырылады.
                 
Эйджизм адамның  жасы бойынша кемсітуді қамтитын мәтіндер. Бұл мәтіндерде тең негізде өзара әрекеттесуге және белгілі бір жас өлшеміне сәйкес келетін адамдармен ғана жұмыс істеуге дайын екендігі көрінеді. 
Біріккен Ұлттар Ұйымының эйджизм мәселесі туралы жаңа есебіне сәйкес, әлемдегі әрбір екінші адам физикалық және психикалық денсаулығының нашарлауына және егде жастағы адамдардың өмір сүру сапасының төмендеуіне әкелетін эйджистік көзқарастарды ұстанады деп саналады.
Қарастырылатын екінші мәселе - еңбекке қабілетті адамдар мүгедектігі бар адамдардан қалыпты және жоғары деп саналатын кемсітушілік түрі. Эйблизм мүгедектігі бар, созылмалы соматикалық немесе психикалық ауытқулары бар адамдарға әлеуметтік бейімділік пен жүйелік кемсітушілікті қамтиды. Ол адамдарды тек шектеулі мүмкіндіктеріне назар аудара отырып сипаттайды және олардың қажеттіліктерін басқа адамдармен салыстырғанда екінші орынға қояды. Осы негізде мүгедектігі бар адамдар белгілі бір дағдыларға немесе мінез-құлық ерекшеліктеріне байланысты немесе керісінше қабылданбайды[5].
Үшінші мәселе –адамның діні, нәсілі, әлеуметтік жағдайы, шығу тегі, ұлтына байланысты кемсіту түрі."""
    )


def page_second():
    st.title("Загрузка данных")
    st.warning("Принимает только файлы формата .csv", icon="⚠️")
    uploaded_file = upload_file()
    if uploaded_file is not None:
        _, file_extension = os.path.splitext(uploaded_file.name)
        if file_extension.lower() == ".csv":
            try:
                df = pd.read_csv(uploaded_file)
                if "original_data" not in st.session_state:
                    st.session_state["original_data"] = df
                else:
                    st.session_state["original_data"] = df
                if "uploaded" not in st.session_state:
                    st.session_state["uploaded"] = True
            except Exception as e:
                st.error(f"Произошла ошибка при чтении файла: {e}", icon="🚨")
        else:
            st.error("Загруженный файл не соответствует формату CSV", icon="🚨")


def page_third():
    if "uploaded" not in st.session_state:
        st.info("Сначала вы должны загрузить данные")
    else:
        if "data" not in st.session_state:
            st.session_state["data"] = st.session_state["original_data"].copy()
        df = st.session_state["original_data"].copy()
        st.title("1. Редактирование данных")
        st.dataframe(df, hide_index=True)
        
        columns = list(df.columns)
        # Выбор колонки для предобработки
        column_for_preprocessing = st.selectbox(
        "Выберите параметры для редактирования",
        options=columns,
        key=persist("column_for_preprocessing"))

        # Обработка данных
        change_df = df
        change_df = process_dataframe(change_df, column_for_preprocessing)

        # Отображение обработанных данных
        st.dataframe(change_df, hide_index=True)

        st.title("2.Визуализация данных")

        color_theme_list = [
            "Blues",
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "Greys",
            "Purples",
            "Greens",
            "Oranges",
            "Reds",
            "YlOrBr",
            "YlOrRd",
            "OrRd",
            "PuRd",
            "RdPu",
            "BuPu",
            "GnBu",
            "PuBu",
            "YlGnBu",
            "PuBuGn",
            "BuGn",
            "YlGn",
            "PiYG",
            "PRGn",
            "BrBG",
            "PuOr",
            "RdGy",
            "RdBu",
            "RdYlBu",
            "RdYlGn",
            "Spectral",
            "coolwarm",
            "bwr",
            "seismic",
        ]
        plots = [
            "Line Plot",
            "Scatter Plot",
            "Histogram Plot",
            "Box Plot",
            "Density Plot",
        ]
        visualizations = st.selectbox(
            "Выберите график который вы хотите увидеть",
            plots,
            key=persist("select_type_of_visualization_for_dataset"),
        )
        x_axis = st.selectbox(
            "Выберите параметр который будет расположен по оси X",
            change_df.columns,
            key=persist("select_column_for_x_axis"),
        )
        y_axis = st.selectbox(
            "Выберите параметр который будет расположен по оси Y",
            change_df.columns,
            key=persist("select_column_for_y_axis"),
        )
        color_theme_list = [
            "blues",
            "cividis",
            "greens",
            "inferno",
            "magma",
            "plasma",
            "reds",
            "turbo",
            "viridis",
        ]

        use_hue = create_toggle(
            "use_hue_toggle", "Включить группировку точек данных по цвету"
        )

        if use_hue:
            hue = st.selectbox(
                "Выберите параметр, с помощью которого будут группироваться точки данных по цвету",
                change_df.columns,
                key="select_hue",  # Assuming persist() function is defined elsewhere to handle session state
            )
            selected_color_theme = st.selectbox(
                "Select a color theme",
                color_theme_list,
                key=persist("select_color_theme"),
            )
        else:
            hue = None
            selected_color_theme = st.color_picker(
                "Pick A Color", "#00f900", key=persist("color")
            )

        # Then, depending on the type of visualization selected by the user:
        if visualizations == "Line Plot":
            line_chart(
                change_df,
                x_axis,
                y_axis,
                input_color_theme=selected_color_theme,
                hue=hue,
            )
        elif visualizations == "Scatter Plot":
            scatter_plot(
                change_df,
                x_axis,
                y_axis,
                input_color_theme=selected_color_theme,
                hue=hue,
            )
        elif visualizations == "Histogram Plot":
            count_of_bins = st.select_slider(
                "Выберите количество колонок", range(2, 100)
            )
            histogram(
                change_df,
                x_axis,
                input_color_theme=selected_color_theme,
                hue=hue,
                bin_count=count_of_bins,
            )
        elif visualizations == "Box Plot":
            box_plot(
                change_df,
                x_axis,
                y_axis,
                input_color_theme=selected_color_theme,
                hue=hue,
            )
        elif visualizations == "Density Plot":
            density_plot(
                change_df, x_axis, input_color_theme=selected_color_theme, hue=hue
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Сохранить изменения"):
                st.session_state["data"] = change_df
                st.success("Изменения сохранены.")
        with col2:
            if st.button("Очистить изменения"):
                st.session_state["data"] = st.session_state["original_data"].copy()
                reset_application_state()
                st.warning("Изменения сброшены")


def page_forth():
    if "data" not in st.session_state:
        st.info("Загрузите данные")
    else:
        complete_df = st.session_state["data"]
        st.title("1. Подготовка датасета для тренировки модели")
        st.dataframe(complete_df, hide_index=True)
        
        # Ensure the persisted value is valid
        if "select_x_column" in st.session_state:
            if st.session_state["select_x_column"] not in complete_df.columns:
                st.session_state["select_x_column"] = complete_df.columns[0]

        if "select_y_column" in st.session_state:
            if st.session_state["select_y_column"] not in complete_df.columns:
                st.session_state["select_y_column"] = complete_df.columns[0]

        x_column = st.selectbox(
            "Выберите `X` столбец с обработанным предложениями:",
            complete_df.columns,
            key=persist("select_x_column"),
        )
        y_column = st.selectbox(
            "Выберите `Y` целевой параметр, значение которого будут спрогнозированы:",
            complete_df.columns,
            key=persist("select_y_column"),
        )
        X = complete_df[x_column]
        y = complete_df[y_column]
        col1, col2, col3 = st.columns([5, 0.3, 1])
        with col1:
            st.subheader("X:")
            st.dataframe(X, hide_index=True)
            st.write("Размер датафрейма :", X.shape)
        with col3:
            st.subheader("y:")
            st.dataframe(y, hide_index=True)
            st.write("Размер датафрейма :", y.shape[0])

        size = st.slider(
            "Выберите размер тестового датафрейма:",
            0.0,
            1.0,
            0.2,
            step=0.01,
            key=persist("slider_size_of_test_dataset"),
        )
        st.write(f"Размер тестового датафрейма {size * 100} % от общего датафрейма")
        X_train, X_test, y_train, y_test = custom_train_test_split(X, y, size=size)

        col1, col2, col3 = st.columns([5, 0.3, 1])
        with col1:
            st.subheader("X_train:")
            st.dataframe(X_train, hide_index=True)
            st.write("Размер датафрейма :", X_train.shape)
        with col3:
            st.subheader("y_train:")
            st.dataframe(y_train, hide_index=True)
            st.write("Размер датафрейма :", y_train.shape[0])
        st.divider()
        col4, col5, col6 = st.columns([5, 0.3, 1])
        with col4:
            st.subheader("X_test:")
            st.dataframe(X_test, hide_index=True)
            st.write("Размер датафрейма :", X_test.shape)
        with col6:
            st.subheader("y_test:")
            st.dataframe(y_test, hide_index=True)
            st.write("Размер датафрейма :", y_test.shape[0])

        st.info("Обязательно нужно активировать векторизацию!")
        vectorizer = create_toggle("vectorizer_toggle", "Активировать векторизацию текста")
        if vectorizer:
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_train_vectors = tfidf_vectorizer.fit_transform(X_train)
            tfidf_test_vectors = tfidf_vectorizer.transform(X_test)

            st.write(f"X_train = {tfidf_train_vectors}")
            st.write(f"X_test = {tfidf_test_vectors}")

        st.title("2. Тренировка и проверка модели")
        classification_models = [
            "Decision Tree CLassification",
            "Random Forest Classification",
            "Support Vector Machine CLassification",
            "Multi Layer Perceptron Classifier",
        ]

        options_clf = st.selectbox(
                "Выберите один из алгоритмов Классификации: ",
                classification_models,
                key=persist("select_classification_model"),
            )
            
        if options_clf == "Decision Tree CLassification":
            options_decision_tree_classification = st.radio(
                "Сделайте выбор:",
                ["Выбрать параметры вручную", "Выбрать дефолтную модель"],
                key=persist("choose_parameters_for_decision_tree_classification"),
            )
            if options_decision_tree_classification == "Выбрать параметры вручную":
                col1, col2 = st.columns(2)
                with col1:
                    max_depth = st.number_input(
                        "`max_depth` Максимальная глубина дерева",
                        1,
                        100,
                        3,
                        key=persist(
                            "number_for_decision_tree_classification_max_depth"
                        ),
                    )
                with col2:
                    min_samples_split = st.number_input(
                        "`min_samples_split` Минимальное количество образцов, необходимое для разбиения внутреннего узла",
                        2,
                        100,
                        2,
                        key=persist(
                            "number_for_decision_tree_classification_min_samples_split"
                        ),
                    )
                col3, col4 = st.columns(2)
                with col3:
                    min_samples_leaf = st.number_input(
                        "`min_samples_leaf` Минимальное количество образцов, необходимое для нахождения в узле листа",
                        1,
                        100,
                        1,
                        key=persist(
                            "number_for_decision_tree_classification_min_samples_leaf"
                        ),
                    )
                with col4:
                    ccp_alpha = st.slider(
                        "`ccp_alpha` Параметр сложности, используемый для обрезки по принципу минимальная стоимость - сложность",
                        0.0,
                        1.0,
                        0.0,
                        step=0.01,
                        key=persist("slider_decision_tree_classification_ccp_alpha"),
                    )
                criterion_for_decision_tree_classification = [
                    "gini",
                    "entropy",
                    "log_loss",
                ]
                criterion = st.selectbox(
                    "`criterion` Функция для измерения качества разделения",
                    criterion_for_decision_tree_classification,
                    key=persist("select_criterion_decision_tree_classification"),
                )

                model = custom_decision_tree_classification(
                    criterion_c=criterion,
                    max_depth_c=max_depth,
                    min_samples_split_c=min_samples_split,
                    min_samples_leaf_c=min_samples_leaf,
                    ccp_alpha_c=ccp_alpha,
                )
            else:
                model = DecisionTreeClassifier()

        elif options_clf == "Random Forest Classification":
            options_random_forest_classification = st.radio(
                "Сделайте выбор:",
                ["Выбрать параметры вручную", "Выбрать дефолтную модель"],
                key=persist("choose_parameters_for_random_forest_classification"),
            )
            if options_random_forest_classification == "Выбрать параметры вручную":
                n_estimators = st.slider(
                    "`n_estimators` Количество деревьев в лесу",
                    50,
                    1000,
                    100,
                    key=persist("number_for_random_forest_classification_n_estimators"),
                )
                col1, col2 = st.columns(2)
                with col1:
                    max_depth = st.number_input(
                        "`max_depth` Максимальная глубина каждого дерева",
                        1,
                        100,
                        3,
                        key=persist(
                            "number_for_random_forest_classification_max_depth"
                        ),
                    )
                with col2:
                    min_samples_split = st.number_input(
                        "`min_samples_split` Минимальное количество образцов, необходимое для разбиения внутреннего узла",
                        2,
                        100,
                        2,
                        key=persist(
                            "number_for_random_forest_classification_min_samples_split"
                        ),
                    )
                col3, col4 = st.columns(2)
                with col3:
                    min_samples_leaf = st.number_input(
                        "`min_samples_leaf` Минимальное количество образцов, необходимое для нахождения в узле листа",
                        1,
                        100,
                        1,
                        key=persist(
                            "number_for_random_forest_classification_min_samples_leaf"
                        ),
                    )
                with col4:
                    ccp_alpha = st.slider(
                        "`ccp_alpha` Параметр сложности, используемый для обрезки по принципу минимальная стоимость - сложность",
                        0.0,
                        1.0,
                        0.0,
                        step=0.01,
                        key=persist("slider_random_forest_classification_ccp_alpha"),
                    )
                criterion_for_random_forest_classification = [
                    "gini",
                    "entropy",
                    "log_loss",
                ]
                criterion = st.selectbox(
                    "`criterion` Функция для измерения качества разделения",
                    criterion_for_random_forest_classification,
                    key=persist("select_criterion_random_forest_classification"),
                )

                model = custom_random_forest_classification(
                    n_estimators,
                    criterion,
                    max_depth,
                    min_samples_split,
                    min_samples_leaf,
                    ccp_alpha,
                )
            else:
                model = RandomForestClassifier()

        elif options_clf == "Support Vector Machine CLassification":
            options_svm_classification = st.radio(
                "Сделайте выбор:",
                ["Выбрать параметры вручную", "Выбрать дефолтную модель"],
                key=persist("choose_parameters_for_svm_classification"),
            )
            if options_svm_classification == "Выбрать параметры вручную":
                col1, col2 = st.columns(2)
                with col1:
                    kernel = st.selectbox(
                        "`kernel` Указывает тип ядра, который будет использоваться в алгоритме.",
                        ["linear", "poly", "rbf", "sigmoid", "precomputed"],
                        key=persist("select_kernel_svm_classification"),
                    )
                with col2:
                    degree = st.number_input(
                        "`degree` Степень полиномиальной функции ядра ('poly')",
                        1,
                        10,
                        3,
                        key=persist("number_for_svm_classification_degree"),
                    )
                model = custom_svc(kernel, degree)
            else:
                model = SVC()
        else:
            options_mlp_classification = st.radio(
                "Сделайте выбор:",
                ["Выбрать параметры вручную", "Выбрать дефолтную модель"],
                key=persist("choose_parameters_for_mlpc"),
            )
            if options_mlp_classification == "Выбрать параметры вручную":
                col1, col2 = st.columns(2)
                with col1:
                    activation = st.selectbox("`activation` Функция активации для скрытого слоя", ["relu", "logistic", "tanh", "identity"], key=persist("select_activation_mlpc"))
                with col2:
                    solver = st.selectbox("`solver` Решающая программа для оптимизации веса", ["adam", "lbfgs", "sgd"],  key=persist("select_solver_mlpc"))
                model = custom_mlp_classifier(activation_func_c=activation, solver_c=solver)
            else:
                model = MLPClassifier()

        st.write(model)

        color_theme_list = [
            "Blues",
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "Greys",
            "Purples",
            "Greens",
            "Oranges",
            "Reds",
            "YlOrBr",
            "YlOrRd",
            "OrRd",
            "PuRd",
            "RdPu",
            "BuPu",
            "GnBu",
            "PuBu",
            "YlGnBu",
            "PuBuGn",
            "BuGn",
            "YlGn",
            "PiYG",
            "PRGn",
            "BrBG",
            "PuOr",
            "RdGy",
            "RdBu",
            "RdYlBu",
            "RdYlGn",
            "Spectral",
            "coolwarm",
            "bwr",
            "seismic",
        ]
        show_results_visualizations = False

        if st.checkbox("Обучить и проверить модель", key=persist("checkbox_model_see")):
            model, pred = train_and_predict(model, tfidf_train_vectors, y_train, tfidf_test_vectors)
            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, average='macro')  
            precision = precision_score(y_test, pred, average='macro')  
            recall = recall_score(y_test, pred, average='macro')
            classification_score = pd.DataFrame(
                {
                    "Accuracy": acc,
                    "F1_score": f1,
                    "Precision": precision,
                    "Recall": recall,
                },
                index=[0],
            )
            st.write("Результаты:")
            show_results_visualizations = True
            st.dataframe(classification_score, hide_index=True)

        # Assuming your model is named `model` and is already trained
        filename = "Completed_model.joblib"
        joblib.dump(model, filename)  # Save the model to a file

        # Open the file in binary mode to pass to the download button
        with open(filename, "rb") as file:
            st.download_button(
                label="Cкачать готовую модель",
                data=file,
                file_name=filename,
                mime="application/octet-stream",
            )

        st.title("3. Визуализация результатов")
        if show_results_visualizations:
            color = st.selectbox("Выберите схему цветов для визуализации", color_theme_list)
            confusion_matrix_visualization(y_test, pred, color)
        else:
            st.info("Сначала нужно обучить модель и получить результаты")
        
        st.title("4. Пример работы с моделью")
        st.image("107169631-1671636988973-gettyimages-1245766411-porzycki-elonmusk221221_npQVV.jpeg")
        message = st.text_input("Введите предложение", key=persist("message_input"))
        if len(message) == 0:
            st.info("Введите предложение") 
        else:
            prediction = model.predict(tfidf_vectorizer.transform([message]))
            st.write(f"Result = {prediction}")





_PERSIST_STATE_KEY = f"{__name__}_PERSIST"


def persist(key: str) -> str:
    """Mark widget state as persistent."""
    if _PERSIST_STATE_KEY not in _state:
        _state[_PERSIST_STATE_KEY] = set()

    _state[_PERSIST_STATE_KEY].add(key)

    return key


initial_state = {"uploaded": True}

initial_state_data = {"uploaded": False}


def load_widget_state():
    """Load persistent widget state."""
    if _PERSIST_STATE_KEY in _state:
        _state.update(
            {
                key: value
                for key, value in _state.items()
                if key in _state[_PERSIST_STATE_KEY]
            }
        )


def reset_application_state():
    global _state
    # Сохранение значений, которые не должны быть удалены
    preserved_values = {
        key: _state[key] for key in ["data", "original_data"] if key in _state
    }

    # Очистка текущего состояния
    _state.clear()

    # Загрузка исходного состояния
    _state.update(
        initial_state.copy()
    )  # Используем .copy() для избежания изменений в исходном словаре

    # Восстановление сохраненных значений
    _state.update(preserved_values)
    st.rerun()


def reset_application_state_with_data():
    global _state

    _state.update(
        initial_state_data.copy()
    )  # Используем .copy() для избежания изменений в исходном словаре

    _state.clear()

    st.rerun()



def upload_file():
    """
    Presents a file uploader widget and returns the uploaded file object.

    This function creates a file uploader in a Streamlit app with the specified prompt message.
    Users can upload a file through the UI. If a file is uploaded, the function returns the
    file object provided by Streamlit, allowing further processing of the file. If no file is
    uploaded, the function returns None.

    Returns:
    - uploaded_file (UploadedFile or None): The file object uploaded by the user through the
      file uploader widget. The object contains methods and attributes to access and read the file's
      content. Returns None if no file has been uploaded.

    Example usage:
    ```
    uploaded_file = upload_file()
    if uploaded_file is not None:
        # Process the file
        st.write("File uploaded:", uploaded_file.name)
    else:
        st.write("No file uploaded.")
    ```
    """
    uploaded_file = st.file_uploader("Загрузите файл")
    if uploaded_file is not None:
        return uploaded_file
    return None


def line_chart(data, x_axis, y_axis, input_color_theme, hue=None):
    # If hue is provided, color the circles based on the hue column and the input color theme.
    # If hue is None, all circles will be colored red.
    if hue:
        color_encoding = alt.Color(hue, scale=alt.Scale(scheme=input_color_theme))
    else:
        color_encoding = alt.value(
            input_color_theme
        )  # Directly set the color to red if hue is not provided

    c = (
        alt.Chart(data)
        .mark_line()
        .encode(x=x_axis, y=y_axis, color=color_encoding)
        .interactive()
    )

    st.altair_chart(c, use_container_width=True)


def scatter_plot(data, x_axis, y_axis, input_color_theme, hue=None):
    # If hue is provided, color the circles based on the hue column and the input color theme.
    # If hue is None, all circles will be colored red.
    if hue:
        color_encoding = alt.Color(hue, scale=alt.Scale(scheme=input_color_theme))
    else:
        color_encoding = alt.value(
            input_color_theme
        )  # Directly set the color to red if hue is not provided

    c = (
        alt.Chart(data)
        .mark_circle()
        .encode(x=x_axis, y=y_axis, color=color_encoding)
        .interactive()
    )

    st.altair_chart(c, use_container_width=True)


import altair as alt
import streamlit as st


def histogram(data, x_axis, input_color_theme, bin_count, hue=None):
    """
    Create a histogram with configurable number of bins (columns).

    :param data: pd.DataFrame - The data to visualize.
    :param x_axis: str - The column name to use for the x-axis.
    :param input_color_theme: str - The color theme for the bars.
    :param bin_count: int - The number of bins (columns) for the histogram.
    :param hue: str or None - The column name for color encoding, if any.
    """
    # If hue is provided, color the bars based on the hue column and the input color theme.
    # If hue is None, all bars will be colored with the input color theme.
    if hue:
        color_encoding = alt.Color(hue, scale=alt.Scale(scheme=input_color_theme))
    else:
        color_encoding = alt.value(
            input_color_theme
        )  # Directly set the color if hue is not provided

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(x_axis, bin=alt.Bin(maxbins=bin_count)),  # Set the number of bins
            y="count()",
            color=color_encoding,
        )
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


def box_plot(data, x_axis, y_axis, input_color_theme, hue=None):
    # If hue is provided, color the box plot based on the hue column and the input color theme.
    # If hue is None, all plots will be colored with the input color theme.
    if hue:
        color_encoding = alt.Color(hue, scale=alt.Scale(scheme=input_color_theme))
    else:
        color_encoding = alt.value(
            input_color_theme
        )  # Directly set the color to red if hue is not provided

    chart = (
        alt.Chart(data)
        .mark_boxplot()
        .encode(x=x_axis, y=y_axis, color=color_encoding)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


def density_plot(data, x_axis, input_color_theme, hue=None):
    # Start with the base chart
    chart = alt.Chart(data)

    # Apply the density transformation conditionally based on hue
    if hue:
        # If hue is provided, calculate density with grouping
        chart = chart.transform_density(
            density=x_axis,
            as_=[x_axis, "density"],
            groupby=[hue],  # Ensure this is a list
        )
    else:
        # If hue is not provided, calculate density without grouping
        chart = chart.transform_density(density=x_axis, as_=[x_axis, "density"])

    # Apply the rest of the encoding
    chart = chart.mark_area().encode(
        x=f"{x_axis}:Q",
        y="density:Q",
        color=(
            alt.Color(hue, scale=alt.Scale(scheme=input_color_theme))
            if hue
            else alt.value(input_color_theme)
        ),
    )

    st.altair_chart(chart, use_container_width=True)


def create_toggle(session_name: str, name: str, status=False) -> bool:
    """
    Creates a toggle switch in a Streamlit app and manages its state across reruns.

    This function displays a toggle switch with the label specified by the 'name' parameter.
    It uses Streamlit's session state to keep track of the toggle's position (on/off) across reruns.
    The current state of the toggle is stored in `st.session_state` using a key provided by the
    'session_name' parameter.

    Parameters:
    - session_name (str): The key used to store the toggle's state in `st.session_state`.
                          This key should be unique to each toggle to prevent state conflicts.
    - name (str): The label displayed next to the toggle switch in the UI.

    Returns:
    - bool: The current state of the toggle (True for on, False for off).

    Example:
    ```
    delete_column = create_toggle('delete_status', 'Удалить')
    if delete_column:
        st.write("Toggle is ON - deletion logic can be placed here.")
    else:
        st.write("Toggle is OFF - no deletion occurs.")
    ```
    """
    # Initialize toggle status in session_state if it doesn't exist
    if session_name not in st.session_state:
        st.session_state[session_name] = False

    # Display the toggle and assign its current value based on session_state
    toggle = st.toggle(name, value=st.session_state[session_name], disabled=status)

    # Update session state based on the toggle's position
    if toggle != st.session_state[session_name]:
        st.session_state[session_name] = toggle
        st.rerun()
    return toggle


def clean_text(text):
    """
    Cleans the input text by removing tabs, extra spaces, punctuation, and converting to lowercase.

    :param text: str - The input text to be cleaned.
    :return: str - The cleaned text.
    """
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"\t+", " ", text)  # Заменяет табуляции на пробелы
    text = re.sub(r"\s+", " ", text)  # Удаляет лишние пробелы
    text = text.strip()  # Убирает пробелы в начале и конце строки
    text = text.lower()  # Преобразует текст в нижний регистр
    punctuationfree = "".join(
        [i for i in text if i not in string.punctuation]
    )  # Убирает пунктуацию
    return punctuationfree


def count_words(text):
    """
    Counts the number of words in the input text.

    :param text: str - The input text to be counted.
    :return: int - The word count.
    """
    if not isinstance(text, str):
        text = str(text)
    words = text.split()
    return len(words)


def process_dataframe(data, column_name):
    """
    Cleans the text in the specified column and adds a 'Word Count' column to the DataFrame.

    :param data: pd.DataFrame - The DataFrame containing the sentences.
    :param column_name: str - The name of the column containing the sentences.
    :return: pd.DataFrame - The DataFrame with cleaned text and an added 'Word Count' column.
    """
    # Очистка текста в указанной колонке
    data[column_name] = data[column_name].apply(clean_text)
    # Добавление колонки с количеством слов
    data["Word Count"] = data[column_name].apply(count_words)
    return data


def delete_columns(data, columns):
    """
    Removes specified columns from a DataFrame.

    This function takes a DataFrame and a list of column names to be removed. It returns a new DataFrame with the specified
    columns removed, leaving the original DataFrame unchanged.

    Parameters:
    - data (pd.DataFrame): The original DataFrame from which columns will be removed.
    - columns (list of str): A list of strings representing the names of the columns to be removed.

    Returns:
    - pd.DataFrame: A new DataFrame with the specified columns removed.
    """
    dataframe = data.drop(columns, axis=1)
    return dataframe


def custom_train_test_split(X, y, size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size)
    return X_train, X_test, y_train, y_test


def custom_decision_tree_classification(
    criterion_c="gini",
    max_depth_c=None,
    min_samples_split_c=2,
    min_samples_leaf_c=2,
    ccp_alpha_c=0.0,
):
    model = DecisionTreeClassifier(
        criterion=criterion_c,
        max_depth=max_depth_c,
        min_samples_split=min_samples_split_c,
        min_samples_leaf=min_samples_leaf_c,
        ccp_alpha=ccp_alpha_c,
    )
    return model


def train_and_predict(model, X_train, y_train, X_test):
    """
    Trains a model and predicts outcomes for the given test data.
    This function uses Streamlit's memoization to cache the model training and prediction steps,
    improving performance for repeated calls with unchanged data.

    Args:
        model: The machine learning model to be trained. Must be hashable by Streamlit.
        X_train (pd.DataFrame or np.ndarray): Training data features.
        y_train (pd.Series or np.ndarray): Training data labels/targets.
        X_test (pd.DataFrame or np.ndarray): Test data features.

    Returns:
        A tuple containing:
        - model: The trained machine learning model.
        - pred: Predictions made by the model on X_test.
    """
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, pred


def custom_random_forest_classification(
    n_estimators_c=100,
    criterion_c="squared_error",
    max_depth_c=None,
    min_samples_split_c=2,
    min_samples_leaf_c=2,
    ccp_alpha_c=0.0,
):
    model = RandomForestClassifier(
        n_estimators=n_estimators_c,
        max_depth=max_depth_c,
        min_samples_split=min_samples_split_c,
        min_samples_leaf=min_samples_leaf_c,
        ccp_alpha=ccp_alpha_c,
    )
    return model


def custom_svc(kernel_c="rbf", degree_c=3):
    model = SVC(kernel=kernel_c, degree=degree_c)
    return model


def custom_mlp_classifier(
    activation_func_c="relu",
    solver_c="adam",
):
    model = MLPClassifier(activation=activation_func_c, solver=solver_c)
    return model


def confusion_matrix_visualization(y_true, y_pred, input_color="Blues"):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=input_color, cbar=False)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    st.pyplot(fig)


if __name__ == "__main__":
    load_widget_state()
    main()
