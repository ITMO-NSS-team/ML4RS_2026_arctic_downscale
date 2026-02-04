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



def plot_metrics(train_metrics, test_metrics, title="График потерь", window=5, metric = "Loss (MSE)"):
    """Построение графиков train и test метрик по эпохам"""
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(train_metrics) + 1)
    if metric == "SSIM":
        epochs = [x * 5 for x in epochs]

    plt.plot(epochs, train_metrics, 'b-', linewidth=2, marker='o', markersize=4, label=f'Train {metric}')
    plt.plot(epochs, test_metrics, 'r-', linewidth=2, marker='s', markersize=4, label=f'Val {metric}')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Эпоха", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(f'convergence.png')
    plt.show()


def visualize_results(lr_images, hr_images, pred_images, epoch='None', model_name="Model", save_path=None):
    """
    Визуализация результатов: LR, Prediction, HR
    """
    n_samples = min(4, len(lr_images))

    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        lr = lr_images[i].squeeze().numpy()
        axes[i, 0].imshow(lr, cmap='Blues', vmin=0, vmax=1)
        axes[i, 0].set_title(f"LR Input (Frame {i + 1})")
        axes[i, 0].axis('off')

        pred = pred_images[i].squeeze().numpy()
        axes[i, 1].imshow(pred, cmap='Blues', vmin=0, vmax=1)
        axes[i, 1].set_title(f"Prediction (Frame {i + 1})")
        axes[i, 1].axis('off')

        hr = hr_images[i].squeeze().numpy()
        axes[i, 2].imshow(hr, cmap='Blues', vmin=0, vmax=1)
        axes[i, 2].set_title(f"Ground Truth HR (Frame {i + 1})")
        axes[i, 2].axis('off')

    plt.suptitle(f"{model_name} - Results (Epoch {epoch})", fontsize=16)
    plt.tight_layout()

    if save_path is None:
        save_path = f"unet_results_epoch_{epoch}.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    return save_path