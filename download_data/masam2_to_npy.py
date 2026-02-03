import xarray as xr
import numpy as np
import os
import tempfile
import shutil
import warnings

warnings.filterwarnings('ignore')


class MASAM2_to_NPY_Converter:
    """
    Конвертирует файлы MASAM2 (.nc.gz) в формат .npy (один файл на день)
    """

    def __init__(self, masam2_dir, output_dir=None, temp_dir=None):
        """
        Args:
            masam2_dir: Папка с файлами MASAM2
            output_dir: Папка для сохранения .npy файлов
            temp_dir: Папка для временных файлов (по умолчанию системная temp)
        """
        self.masam2_dir = masam2_dir
        if output_dir is None:
            self.output_dir = os.path.join(os.path.dirname(masam2_dir), 'MASAM2_npy')
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        if temp_dir is None:
            self.temp_dir = os.path.join(tempfile.gettempdir(), 'masam2_temp')
        else:
            self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

        self.stats = {
            'total_files': 0,
            'total_frames': 0,
            'converted_frames': 0,
            'failed_files': []
        }

    def read_masam2_file(self, file_path):
        """
        Читает файл MASAM2, поддерживая .gz сжатие
        Возвращает xarray Dataset
        """
        print(f"Чтение файла: {os.path.basename(file_path)}")

        try:
            # Пробуем разные движки для чтения
            engines_to_try = ['h5netcdf', 'netcdf4']

            for engine in engines_to_try:
                try:
                    ds = xr.open_dataset(file_path, engine=engine)
                    return ds
                except:
                    continue

            temp_file = os.path.join(self.temp_dir, os.path.basename(file_path).replace('.gz', ''))
            if file_path.endswith('.gz'):
                import gzip
                with gzip.open(file_path, 'rb') as f_in:
                    with open(temp_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                for engine in engines_to_try:
                    try:
                        ds = xr.open_dataset(temp_file, engine=engine)
                        self._temp_file_to_delete = temp_file
                        return ds
                    except:
                        continue

                if os.path.exists(temp_file):
                    os.remove(temp_file)

            raise Exception("Не удалось открыть файл ни одним из способов")

        except Exception as e:
            print(f"Ошибка чтения файла: {e}")
            return None


    def convert_masam2_file(self, file_path, output_subdir=None):
        """
        Конвертирует один файл MASAM2 в несколько .npy файлов (по одному на день)
        """
        print(f"\n{'=' * 60}")
        print(f"КОНВЕРТАЦИЯ: {os.path.basename(file_path)}")
        print(f"{'=' * 60}")

        ds = None
        try:
            ds = self.read_masam2_file(file_path)
            if ds is None:
                self.stats['failed_files'].append(os.path.basename(file_path))
                return

            if 'Sea_Ice_Concentration' not in ds:
                print("В файле нет переменной 'sea ice concentration'")
                self.stats['failed_files'].append(os.path.basename(file_path))
                return

            ice_data = ds['Sea_Ice_Concentration'].values
            ice_data = np.rot90(ice_data, k=1, axes=(1, 2))
            ice_data = np.flip(ice_data, axis=1)

            if 'Day of Month' in ds:
                days = ds['Day of Month'].values
            else:
                days = np.arange(1, ice_data.shape[0] + 1)

            print(f"Данные загружены:")
            print(f"   Форма: {ice_data.shape} (дней×высота×ширина)")
            print(f"   Тип данных: {ice_data.dtype}")
            print(f"   Дни: {days}")

            file_name = os.path.basename(file_path)
            parts = file_name.split('.')
            year_month = parts[1]
            year = year_month[:4]
            month = year_month[4:6]


            self.stats['total_files'] += 1
            self.stats['total_frames'] += ice_data.shape[0]

            if output_subdir is None:
                base_name = f"{year}_{month}"
                output_subdir = os.path.join(self.output_dir, base_name)
            else:
                output_subdir = os.path.join(self.output_dir, output_subdir)

            os.makedirs(output_subdir, exist_ok=True)

            print("\n Сохранение файлов .npy...")

            for i, day_num in enumerate(days):
                day_data = ice_data[i]

                day_data_processed = self.process_values(day_data)

                day_str = f"{int(day_num):02d}"
                output_filename = f"masam2_{year}{month}{day_str}.npy"
                output_path = os.path.join(output_subdir, output_filename)
                np.save(output_path, day_data_processed)

                all_days_dir = os.path.join(self.output_dir, 'all_days')
                os.makedirs(all_days_dir, exist_ok=True)
                all_days_path = os.path.join(all_days_dir, output_filename)
                np.save(all_days_path, day_data_processed)

                print(f"   День {day_str}: {output_filename} ({day_data_processed.shape}, {day_data_processed.dtype})")
                self.stats['converted_frames'] += 1

            print(f"\n Файл успешно конвертирован!")
            print(f"   Файлы сохранены в: {output_subdir}")
            print(f"   Также в: {os.path.join(self.output_dir, 'all_days')}")

        except Exception as e:
            print(f" Ошибка конвертации файла {file_path}: {e}")
            import traceback
            traceback.print_exc()
            self.stats['failed_files'].append(os.path.basename(file_path))

        finally:
            if ds is not None:
                ds.close()

            if hasattr(self, '_temp_file_to_delete') and os.path.exists(self._temp_file_to_delete):
                try:
                    os.remove(self._temp_file_to_delete)
                    delattr(self, '_temp_file_to_delete')
                except:
                    pass

    def process_values(self, data):
        cleaned = data.copy().astype(np.int16)
        cleaned[(cleaned == 104) | (cleaned == 119) | (cleaned == 120)] = 0
        cleaned = np.clip(cleaned, 0, 100)
        processed = np.round(cleaned).astype(np.uint8)

        return processed



    def batch_convert(self, file_pattern='masam2.*.nc'):
        """
        Пакетная конвертация всех файлов MASAM2 в директории
        """
        print(f"\n{'=' * 60}")
        print(f"ПАКЕТНАЯ КОНВЕРТАЦИЯ ФАЙЛОВ MASAM2")
        print(f"Исходная папка: {self.masam2_dir}")
        print(f"Целевая папка: {self.output_dir}")
        print(f"Временная папка: {self.temp_dir}")
        print(f"{'=' * 60}")

        masam2_files = []
        for root, dirs, files in os.walk(self.masam2_dir):
            for file in files:
                if (file.endswith('.nc.gz') or file.endswith('.nc')) and 'masam2' in file.lower():
                    masam2_files.append(os.path.join(root, file))

        print(f" Найдено файлов MASAM2: {len(masam2_files)}")

        if len(masam2_files) == 0:
            print("️  Файлы MASAM2 не найдены!")
            print("   Ожидаемый формат: masam2.YYYYMM.nc.gz или masam2.YYYYMM.nc")
            return


        print(f"\n НАЧИНАЕМ КОНВЕРТАЦИЮ...")

        for i, file_path in enumerate(masam2_files):
            print(f"\n[{i + 1}/{len(masam2_files)}] ", end="")
            self.convert_masam2_file(file_path)



def quick_convert(masam2_file, output_dir=None):
    """
    Быстрая конвертация одного файла MASAM2
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(masam2_file), 'npy_output')

    converter = MASAM2_to_NPY_Converter(
        masam2_dir=os.path.dirname(masam2_file),
        output_dir=output_dir
    )
    converter.convert_masam2_file(masam2_file)

    npy_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.npy') and file.startswith('masam2_'):
                npy_files.append(os.path.join(root, file))

    if npy_files:
        print(f"\nПЕРВЫЙ СОЗДАННЫЙ ФАЙЛ:")
        sample = np.load(npy_files[0])
        print(f"   Имя: {os.path.basename(npy_files[0])}")
        print(f"   Размер: {sample.shape}")
        print(f"   Тип: {sample.dtype}")


# Главная функция
if __name__ == "__main__":
    print("Конвертер MASAM2 → .npy формат")

    for year in ['20' + str(i) for i in range(25, 26)]:
        for month in [str(j) for j in range(1, 3)]:
            if len(month) == 1:
                month = '0' + month

            filename = f"masam2.{year}{month}.nc"
            input_file = f"D:/MASAM2/{year}/{filename}"

            if os.path.exists(input_file):
                print(f"Найден файл: {input_file}")
                output_dir = os.path.join(os.path.dirname(os.path.dirname(input_file)), 'MASAM2_npy')
                print(f" Выходная директория: {output_dir}")

                quick_convert(input_file, output_dir)

            else:
                print(f" Файл не найден: {input_file}")