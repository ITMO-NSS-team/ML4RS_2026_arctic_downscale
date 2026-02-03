import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def visualize_example(file_path):
    if not os.path.exists(file_path):
        print(f"Файл не найден: {file_path}")
    else:
        try:
            data = np.load(file_path)

            print(f"Размер: {data.shape}")
            output_dir = 'example_analysis'
            os.makedirs(output_dir, exist_ok=True)

            plt.figure(figsize=(10, 10))
            plt.imshow(data, cmap='Blues', origin='lower')
            plt.colorbar(label='Концентрация льда (%)')
            plt.title('Data Example')
            plt.savefig(f'{output_dir}/example_plot.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"График сохранен: {output_dir}/example_plot.png")

            plt.figure(figsize=(10, 5))
            plt.hist(data.flatten(), bins=50, alpha=0.7)
            plt.xlabel('Значение')
            plt.ylabel('Частота')
            plt.title('Распределение значений')
            plt.savefig(f'{output_dir}/example_histogram.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Гистограмма сохранена: {output_dir}/example_histogram.png")

            print("\nДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ:")
            print(f"Форма данных: {data.shape}")
            print(f"Общее число пикселей: {data.size}")

        except Exception as e:
            print(f"Ошибка при загрузке файла: {e}")



file_path = "D:/MASAM2/MASAM2_npy/2013_01/masam2_20130101.npy"   #"D:/2013/osisaf_20130101.npy"
visualize_example(file_path)