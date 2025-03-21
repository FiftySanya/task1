# Програма сортування списку цілих чисел за зростанням різними способами на мові C

## Опис
Ця програма реалізує сортування чисел за зростанням за значенням і частотою їх появи у вхідному масиві. Програма підтримує три алгоритми сортування:
- **Швидке сортування (QuickSort)**
- **Сортування злиттям (MergeSort)**
- **Купчасте сортування (HeapSort)**

Крім того, використовується **паралелізація** за допомогою OpenMP, щоб прискорити обчислення при виконанні сортування.

## Компіляція програми
```sh
gcc -fopenmp -Wall -Wextra task1.c -o task1
```

## Використання
```sh
./task1 -t <тип_сортування> -k <ключ_сортування> <список чисел>
```
Де:
- `-t` визначає тип сортування (`qsort`, `merge`, `heap`)
- `-k` визначає ключ сортування (`value-freq` або `freq-value`)
- `<список чисел>` — послідовність чисел для сортування

### Приклад запуску
```sh
./task1 -t qsort -k value-freq 4 1 3 4 2 3 3 1 2
```

## Функції

### `calculate_frequencies(const int arr[], int size, NumberInfo result[])`
Обчислює частоту кожного числа у вхідному масиві та зберігає результати у структурі `NumberInfo`.

### `compare_value_freq(const void *a, const void *b)`
Функція порівняння для сортування за зростанням значення, а при рівних значеннях — за спаданням частоти.

### `compare_freq_value(const void *a, const void *b)`
Функція порівняння для сортування за спаданням частоти, а при рівних частотах — за зростанням значення.

### `partition(NumberInfo arr[], int low, int high, int (*compare_func)(const void *, const void *))`
Допоміжна функція для розбиття масиву для QuickSort.

### `parallel_qsort(NumberInfo arr[], int low, int high, int (*compare_func)(const void *, const void *))`
Реалізація **паралельного QuickSort** з використанням OpenMP.

### `parallel_merge(NumberInfo arr[], int left, int mid, int right, int (*compare_func)(const void *, const void *))`
Допоміжна функція для **паралельного MergeSort**, яка об'єднує два відсортованих підмасиви.

### `parallel_merge_sort(NumberInfo arr[], int left, int right, int (*compare_func)(const void *, const void *))`
Реалізація **паралельного MergeSort** з використанням OpenMP.

### `heapify(NumberInfo arr[], int n, int i, int (*compare_func)(const void *, const void *))`
Перетворює піддерево в коректну купу для HeapSort.

### `heap_sort(NumberInfo arr[], int n, int (*compare_func)(const void *, const void *))`
Реалізація **HeapSort**.

## Роль паралелізації
Програма використовує OpenMP для **прискорення обчислень** шляхом розподілу завдань між кількома потоками. Це дозволяє:
- Виконувати QuickSort на двох підмасивах одночасно
- Прискорювати MergeSort, обробляючи ліву та праву частини масиву в паралельних потоках

## Висновок
Програма дозволяє ефективно сортувати числа за значенням та частотою з використанням різних алгоритмів та паралелізації. OpenMP допомагає значно зменшити час виконання при обробці великих наборів даних.

