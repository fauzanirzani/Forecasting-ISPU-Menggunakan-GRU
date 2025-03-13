import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import datetime

# Konfigurasi halaman Streamlit
st.title("Sistem Forecasting ISPU Menggunakan GRU")
st.sidebar.title("Menu")

# Upload file
uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
if uploaded_file:
    # Membaca data
    data = pd.read_csv(uploaded_file)
    st.write("### Data yang diunggah:")
    st.dataframe(data)

    # Pastikan kolom tersedia
    if 'Tanggal' in data.columns and 'ISPU' in data.columns:
        train_treshold = 0.7  # Batas data latih (70%)
        column = 'ISPU'

        # Membuat DataFrame untuk kolom 'ispu'
        df = pd.DataFrame({'Date': data['Tanggal'], 'data': data[column]})
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Membagi data menjadi data latih dan uji
        st.write("### Pembagian Data Uji dan Data Latih")
        train_size = int(len(df) * train_treshold)
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]
        st.write('Total '+column+' Data Train :', train_data.size)
        st.write('Total '+column+' Data Test :', test_data.size)

        # Normalisasi data
        st.write("### Proses Normalisasi Data")
        scaler = MinMaxScaler().fit(train_data)
        train_scaled = scaler.transform(train_data)
        test_scaled = scaler.transform(test_data)
        # Menampilkan train_data dan test_data secara horizontal
        col1, col2 = st.columns(2)
        with col1:
            st.write("Hasil Sebelum Normalisasi - Train Data")
            st.write(train_data)
        with col2:
            st.write("Hasil Sebelum Normalisasi - Test Data")
            st.write(test_data)

        # Menampilkan train_scaled dan test_scaled secara horizontal
        col3, col4 = st.columns(2)
        with col3:
            st.write("Hasil Normalisasi - Train Scaled")
            st.write(train_scaled)
        with col4:
            st.write("Hasil Normalisasi - Test Scaled")
            st.write(test_scaled)

        # Membuat dataset input
        look_back_step = 7

        def create_dataset(X, look_back=1):
            Xs, ys = [], []
            for i in range(len(X) - look_back):
                v = X[i:i + look_back]
                Xs.append(v)
                ys.append(X[i + look_back])
            return np.array(Xs), np.array(ys)
        
 
        X_train, y_train = create_dataset(train_scaled, look_back_step)
        X_test, y_test = create_dataset(test_scaled, look_back_step)

        st.write("### Proses Windowing")
        st.write('X_train.shape:', X_train.shape)
        st.write('y_train.shape:', y_train.shape)
        st.write('X_test.shape:', X_test.shape)
        st.write('y_test.shape:', y_test.shape)

        

        # Membuat model GRU
        units = 64
        model = Sequential([
            GRU(units=units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]),
                activation='tanh', recurrent_activation='sigmoid'),
            Dropout(0.2),
            GRU(units=units, activation='tanh', recurrent_activation='sigmoid'),
            Dropout(0.2),
            Dense(units=1),
            # Dense(units = 1, activation='tanh')
            # Dense(units = 1, activation='linear')
            # Dense(units = 1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='mse')

        # Melatih model
        # early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        # rlrop = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

        history = model.fit(X_train, y_train, epochs=1000, validation_split=0.1,
                            batch_size=16, shuffle=False)
        
        st.write("### Grafik Model Train vs Validation Loss:")
        plt.figure(figsize = (10, 6))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Train vs Validation Loss for '+column )
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.legend(['Train loss', 'Validation loss'], loc='upper right')
        st.pyplot(plt)

        # Evaluasi model
        prediction = model.predict(X_train)
        st.write(f"Nilai Prediksi Sebelum Denormalisasi: {prediction}")
        prediction = scaler.inverse_transform(prediction)
        y_test = scaler.inverse_transform(y_train)

        # Menghitung metrik evaluasi
        rmse = np.sqrt(mean_squared_error(y_test, prediction))
        mape = mean_absolute_percentage_error(y_test, prediction) * 100
        r2 = r2_score(y_test, prediction)

        # Menampilkan hasil evaluasi
        st.write("### Evaluasi Model:")
        st.write(f"**RMSE:** {rmse:.4f}")
        st.write(f"**MAPE:** {mape:.2f}%")
        st.write(f"**R²:** {r2:.4f}")

        # Menampilkan hasil forecast
        pred = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': prediction.flatten()})
        st.write("### Hasil Prediksi vs Data Aktual:")
        st.dataframe(pred)
        

        # Plot hasil
        st.write("### Grafik Data Prediksi dan Data Aktual")
        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label='Data Aktual')
        plt.plot(prediction, label='Prediksi')
        plt.title('Prediksi vs Data Aktual')
        plt.legend()
        st.pyplot(plt)
        

        st.write("### Hasil Forecasting data selama 7 hari kedepan:")
             # Membuat DataFrame untuk kolom 'ispu'
        df = pd.DataFrame({'Date': data['Tanggal'], 'data': data[column]})
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        next_seven_days = []
        for x in range(7):
                days = x+1
                n_day = df.index[-1] + datetime.timedelta(days=days)
                n_day = n_day.strftime("%Y-%m-%d")
                next_seven_days.append(n_day)
            
        # _ispu = proccessing_data_gru(df, column ,next_seven_days)
        forecasting = pd.DataFrame({'ISPU': prediction.flatten()})
        forecast = forecasting.tail(7)
        print(forecasting)
        result = pd.concat([forecasting], axis=1)
        print(result)
        colors = []
        for _, r in forecast.iterrows():
            predicted_r = r['ISPU']
            if predicted_r <= 50:
                colors.append('#2ecc71')
            elif predicted_r > 50 and predicted_r <= 100:
                colors.append("#3498db")
            elif predicted_r > 100 and predicted_r <= 200:
                colors.append("#f1c40f")
            elif predicted_r > 200 and predicted_r <= 300:
                colors.append("#e74c3c")
            else:
                colors.append("#2c3e50")

        plt.figure(figsize=(10, 6))
        # Membuat bar chart
        bars = plt.bar(next_seven_days, forecast['ISPU'].tolist(), color=colors)

        # Menambahkan keterangan warna di bawah setiap bar
        for bar, color in zip(bars, colors):
            label = ""
            if color == '#2ecc71':
                label = "Baik"
            elif color == "#3498db":
                label = "Sedang"
            elif color == "#f1c40f":
                label = "Tidak Sehat"
            elif color == "#e74c3c":
                label = "Sangat Tidak Sehat"
            else:
                label = "Berbahaya"
            
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, label, 
                    ha='center', va='bottom', fontsize=9, rotation=90)
            
        plt.legend(loc='upper right')
        plt.xlabel('Hari')
        plt.ylabel('Nilai Max')
        plt.xticks(range(7),next_seven_days)
        st.pyplot(plt)

        # Membuat DataFrame baru yang menggabungkan next_seven_days dan forecast
        forecast_with_days = pd.DataFrame({
            'Tanggal': next_seven_days,
            'Prediksi ISPU': forecast['ISPU'].tolist()
})

        # Menampilkan DataFrame di Streamlit
        st.dataframe(forecast_with_days, use_container_width=True)

        # # Setelah Hasil Forecasting 7 Hari Kedepan
        # st.write("### Pengujian Model dengan Berbagai Fungsi Aktivasi dan Parameter")

        # # Menentukan pengaturan untuk pengujian
        # activations = ['linear', 'tanh', 'sigmoid']
        # # learning_rate = 0.001
        # factor = 0.1
        # patience_early_stopping = 15
        # patience_lr = 10

        # # Melakukan pengujian untuk masing-masing aktivasi
        # for activation in activations:
        #     st.write(f"### Menguji Fungsi Aktivasi: {activation}")

        #     # Membangun model GRU untuk setiap fungsi aktivasi
        #     model = Sequential([
        #         GRU(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]),
        #             activation=activation, recurrent_activation='sigmoid'),
        #         Dropout(0.2),
        #         GRU(units=64, activation=activation, recurrent_activation='sigmoid'),
        #         Dropout(0.2),
        #         Dense(units=1),
        #     ])
        #     model.compile(optimizer='adam', loss='mse')

        #     # Callbacks untuk EarlyStopping dan ReduceLROnPlateau
        #     early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_early_stopping)
        #     rlrop = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=patience_early_stopping)

        #     # Melatih model
        #     history = model.fit(X_train, y_train, epochs=500, validation_split=0.1, batch_size=16, 
        #                         shuffle=False)

        #     # Menampilkan grafik loss
        #     st.write(f"### Grafik Loss untuk Fungsi Aktivasi {activation}")
        #     plt.figure(figsize=(10, 6))
        #     plt.plot(history.history['loss'], label='Train Loss')
        #     plt.plot(history.history['val_loss'], label='Validation Loss')
        #     plt.title(f'Model Loss for {activation}')
        #     plt.xlabel('Epoch')
        #     plt.ylabel('Loss')
        #     plt.legend()
        #     st.pyplot(plt)

        #     # Evaluasi model
        #     prediction = model.predict(X_train)
        #     prediction = scaler.inverse_transform(prediction)
        #     y_test = scaler.inverse_transform(y_train)

        #     rmse = np.sqrt(mean_squared_error(y_test, prediction))
        #     mape = mean_absolute_percentage_error(y_test, prediction) * 100
        #     r2 = r2_score(y_test, prediction)

        #     # Menampilkan hasil evaluasi
        #     st.write(f"### Evaluasi Model dengan Aktivasi: {activation}")
        #     st.write(f"**RMSE:** {rmse:.4f}")
        #     st.write(f"**MAPE:** {mape:.2f}%")
        #     st.write(f"**R²:** {r2:.4f}")

        #     # Menampilkan hasil prediksi
        #     st.write(f"### Hasil Prediksi vs Data Aktual untuk Aktivasi {activation}")
        #     pred_df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': prediction.flatten()})
        #     st.dataframe(pred_df)

        #     # Grafik Prediksi vs Data Aktual
        #     st.write(f"### Grafik Prediksi vs Data Aktual untuk Aktivasi {activation}")
        #     plt.figure(figsize=(10, 6))
        #     plt.plot(y_test, label='Data Aktual')
        #     plt.plot(prediction, label='Prediksi')
        #     plt.title(f'Prediksi vs Data Aktual untuk {activation}')
        #     plt.legend()
        #     st.pyplot(plt)

    else:
        st.error("Kolom 'Tanggal' atau 'ISPU' tidak ditemukan dalam file.")
else:
    st.info("Silakan unggah file CSV.")

    #python -m streamlit run TA_152020086_FauzanIrzani_Code.py